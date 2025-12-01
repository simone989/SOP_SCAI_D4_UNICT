import collections
import itertools
import logging
import re
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
from bertopic import BERTopic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


class ThreatEntry(BaseModel):
    threat: str = Field(..., description="Threat label in SVO-like format")
    explanation: str = Field(..., description="High-level rationale for the threat")


class ContextEnum(str, Enum):
    AI_ACT = "ai_act"
    NIS_2 = "nis_2"
    ISO_9241_210 = "iso_9241-210"


class FilterRequest(BaseModel):
    threat: List[ThreatEntry] = Field(..., description="Threat candidates provided by the workflow")
    context: ContextEnum = Field(..., description="Corpus or regulatory context for filtering")


class FilterMetadata(BaseModel):
    threshold: float
    total_threats: int
    retained_threats: int
    mean_score: float
    relevant_topics: List[int]
    relevant_topic_similarities: List[float]


class FilterResponse(BaseModel):
    context: ContextEnum
    hypernym_phrase: str
    filtered_threats: List[ThreatEntry]
    metadata: FilterMetadata
    notes: str


class SVORepresentation(BaseModel):
    subject: str = Field(..., description="Subject component of the threat (noun-like)")
    verb: str = Field(..., description="Verb component of the threat")
    object: str = Field(..., description="Object component of the threat (noun-like)")


class HypernymRanking(BaseModel):
    term: str
    count: int


class HypernymExtractionMetadata(BaseModel):
    total_svos: int
    subject_with_match: int
    verb_with_match: int
    object_with_match: int
    ranked_threats_considered: int


class RoleSimilarity(BaseModel):
    mean_sim: float
    std_sim: float


class ThreatSimilarity(BaseModel):
    subject: Optional[RoleSimilarity] = None
    verb: Optional[RoleSimilarity] = None
    object: Optional[RoleSimilarity] = None
    aggregate_score: float


class RankedThreat(BaseModel):
    original_threat: Optional[Dict[str, str]] = None
    svo_representation: SVORepresentation
    similarity: ThreatSimilarity


class HypernymExtractionResponse(BaseModel):
    subject: List[HypernymRanking]
    verb: List[HypernymRanking]
    object: List[HypernymRanking]
    ranked_threats: List[RankedThreat]
    metadata: HypernymExtractionMetadata


class MergeIteration(BaseModel):
    ranked_threats: List[RankedThreat] = Field(
        default_factory=list,
        description="Ranked threats coming from a single /filter_oracol_1 run.",
    )


class MergeRequest(BaseModel):
    iterations: List[MergeIteration] = Field(
        ..., description="Collection of /filter_oracol_1 iterations to be merged."
    )
    lemmatize_subjects: bool = Field(
        True,
        description="Apply WordNet lemmatization to subject terms before generating combinations.",
    )
    max_combinations: Optional[int] = Field(
        None,
        gt=0,
        description="Optional cap on the number of SVO combinations returned.",
    )


class MergeMetadata(BaseModel):
    total_iterations: int
    unique_subjects: int
    unique_verbs: int
    unique_objects: int
    generated_combinations: int


class MergeResponse(BaseModel):
    combinations: List[SVORepresentation]
    metadata: MergeMetadata


class BodyThreatEntry(BaseModel):
    original_threat: Optional[Dict[str, str]] = None
    svo_representation: SVORepresentation


class HypernymExtractionRequest(BaseModel):
    threat: List[BodyThreatEntry] = Field(
        default_factory=list,
        description="Threat entries already converted to SVO representations",
    )
    svos: List[SVORepresentation] = Field(
        default_factory=list,
        description="Optional direct list of SVO entries; merged with `threat` if present",
    )
    top_k: int = Field(20, ge=1, le=100, description="Top hypernyms to return per role")


app = FastAPI(title="Medium-High Oracle Service", version="0.3.0")


_HYPERNYM_TEMPLATES = {
    ContextEnum.AI_ACT: "An AI threat is any circumstance, event, or agent—human or automated—that, by exploiting or targeting the functions, models, data, or infrastructure of an artificial‑intelligence system throughout its life‑cycle, has the potential to compromise its confidentiality, integrity, availability, traceability, or reliability.",
    ContextEnum.NIS_2: "Any circumstance or event with the potential to adversely impact organizational operations (including mission, functions, image, or reputation), organizational assets, or individuals through an information system via unauthorized access, destruction, disclosure, modification of information, and/or denial of service.",
    ContextEnum.ISO_9241_210: "A human factor threat is any condition in the interactive system, environment, or context of use that hinders the human-centered design principles, negatively impacting the user's ability to achieve goals with effectiveness, efficiency, and satisfaction, or failing to meet accessibility and safety requirements.",
}

WORDNET_RESOURCES = ("wordnet", "omw-1.4")
GENERIC_HYPERNYM_BAN = {
    "entity",
    "object",
    "physical entity",
    "physical object",
    "thing",
    "whole",
    "unit",
}
NO_MATCH_TOKEN = "<no-match>"


def _ensure_wordnet_resources() -> None:
    """Ensure the required WordNet corpora are available locally."""

    for resource in WORDNET_RESOURCES:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource '%s' for WordNet support.", resource)
            nltk.download(resource, quiet=True)


def _normalize_term(term: str) -> str:
    return term.strip().lower()


def _wordnet_synsets(term: str, pos: str):
    from nltk.corpus import wordnet as wn

    return wn.synsets(term.replace(" ", "_"), pos=pos)


def _immediate_hypernyms(term: str, pos: str) -> set:
    hypers = set()
    for syn in _wordnet_synsets(term, pos):
        for hyp in syn.hypernyms():
            hypers |= {lemma.replace("_", " ") for lemma in hyp.lemma_names()}
    return hypers - GENERIC_HYPERNYM_BAN


def _fallback_single_word(term: str, pos: str) -> set:
    words = re.split(r"\s+", term.strip())
    if len(words) <= 1:
        return set()
    return _immediate_hypernyms(words[-1], pos)


def _hypernyms_for_term(term: str, pos: str) -> set:
    hypers = _immediate_hypernyms(term, pos)
    return hypers or _fallback_single_word(term, pos)


def _aggregate_hypernyms(svos: List[SVORepresentation]) -> Tuple[Dict[str, collections.Counter], Dict[str, int]]:
    counters = {
        "subject": collections.Counter(),
        "verb": collections.Counter(),
        "object": collections.Counter(),
    }
    stats = {"subject_matches": 0, "verb_matches": 0, "object_matches": 0}

    for entry in svos:
        subject = _normalize_term(entry.subject)
        verb = _normalize_term(entry.verb)
        obj = _normalize_term(entry.object)

        if subject:
            hypers = _hypernyms_for_term(subject, "n")
            matched = bool(hypers)
            counters["subject"].update(hypers or [NO_MATCH_TOKEN])
            if matched:
                stats["subject_matches"] += 1

        if verb:
            hypers = _hypernyms_for_term(verb, "v")
            matched = bool(hypers)
            counters["verb"].update(hypers or [NO_MATCH_TOKEN])
            if matched:
                stats["verb_matches"] += 1

        if obj:
            hypers = _hypernyms_for_term(obj, "n")
            matched = bool(hypers)
            counters["object"].update(hypers or [NO_MATCH_TOKEN])
            if matched:
                stats["object_matches"] += 1

    return counters, stats


def _collect_entries(payload: HypernymExtractionRequest) -> List[BodyThreatEntry]:
    entries = list(payload.threat)
    entries.extend(
        BodyThreatEntry(original_threat=None, svo_representation=svo) for svo in payload.svos
    )
    return entries


def _collect_terms_from_iterations(payload: MergeRequest) -> Tuple[set, set, set]:
    subjects = set()
    verbs = set()
    objects = set()

    for iteration in payload.iterations:
        for threat in iteration.ranked_threats:
            svo = threat.svo_representation
            if svo.subject:
                subjects.add(svo.subject.strip())
            if svo.verb:
                verbs.add(svo.verb.strip())
            if svo.object:
                objects.add(svo.object.strip())

    return subjects, verbs, objects


def _lemmatize_subjects(subjects: List[str]) -> List[str]:
    if not subjects:
        return []

    _ensure_wordnet_resources()
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    def _lemmatize_phrase(phrase: str) -> str:
        tokens = [token for token in phrase.split() if token]
        return " ".join(lemmatizer.lemmatize(token) for token in tokens) if tokens else ""

    return sorted({
        _lemmatize_phrase(subject.strip()) if subject else ""
        for subject in subjects
        if subject
    })


def _generate_svo_combinations(
    subjects: List[str],
    verbs: List[str],
    objects: List[str],
    limit: Optional[int] = None,
) -> List[SVORepresentation]:
    if not subjects or not verbs or not objects:
        return []

    seen = set()
    combinations: List[SVORepresentation] = []
    for subj, verb, obj in itertools.product(subjects, verbs, objects):
        if not subj or not verb or not obj:
            continue
        key = (subj, verb, obj)
        if key in seen:
            continue
        seen.add(key)
        combinations.append(SVORepresentation(subject=subj, verb=verb, object=obj))
        if limit and len(combinations) >= limit:
            break

    return combinations


def _rank_threats_against_hypernyms(
    entries: List[BodyThreatEntry],
    hypernym_terms: Dict[str, List[str]],
    top_k: int,
) -> List[RankedThreat]:
    roles = ("subject", "verb", "object")

    if not any(hypernym_terms.get(role) for role in roles):
        empty_rankings = [
            RankedThreat(
                original_threat=entry.original_threat,
                svo_representation=entry.svo_representation,
                similarity=ThreatSimilarity(
                    subject=None,
                    verb=None,
                    object=None,
                    aggregate_score=0.0,
                ),
            )
            for entry in entries
        ]
        empty_rankings.sort(key=lambda item: item.similarity.aggregate_score, reverse=True)
        return empty_rankings[:top_k]

    embedder = _get_embedder()

    hypernym_matrices: Dict[str, Optional[np.ndarray]] = {}
    for role in roles:
        normalized_terms = sorted(
            {
                _normalize_term(term)
                for term in hypernym_terms.get(role, [])
                if term and term != NO_MATCH_TOKEN
            }
        )
        if not normalized_terms:
            hypernym_matrices[role] = None
            continue
        vectors = embedder.encode(normalized_terms, convert_to_numpy=True, show_progress_bar=False)
        hypernym_matrices[role] = np.asarray(vectors)

    term_embedding_maps: Dict[str, Dict[str, np.ndarray]] = {role: {} for role in roles}
    for role in roles:
        if hypernym_matrices[role] is None:
            continue
        role_terms = []
        for entry in entries:
            value = getattr(entry.svo_representation, role)
            normalized = _normalize_term(value)
            if normalized:
                role_terms.append(normalized)
        unique_terms = sorted(set(role_terms))
        if not unique_terms:
            continue
        vectors = embedder.encode(unique_terms, convert_to_numpy=True, show_progress_bar=False)
        term_embedding_maps[role] = {
            term: vector for term, vector in zip(unique_terms, np.asarray(vectors))
        }

    def _role_similarity(term_value: str, role: str) -> Optional[RoleSimilarity]:
        matrix = hypernym_matrices.get(role)
        if matrix is None:
            return None
        normalized = _normalize_term(term_value)
        if not normalized:
            return None
        vector = term_embedding_maps.get(role, {}).get(normalized)
        if vector is None:
            return None
        sims = cosine_similarity(vector.reshape(1, -1), matrix)
        return RoleSimilarity(mean_sim=float(np.mean(sims)), std_sim=float(np.std(sims)))

    ranked = []
    for entry in entries:
        role_stats = {role: _role_similarity(getattr(entry.svo_representation, role), role) for role in roles}
        available_means = [stat.mean_sim for stat in role_stats.values() if stat is not None]
        aggregate = float(np.mean(available_means)) if available_means else 0.0
        ranked.append(
            RankedThreat(
                original_threat=entry.original_threat,
                svo_representation=entry.svo_representation,
                similarity=ThreatSimilarity(
                    subject=role_stats["subject"],
                    verb=role_stats["verb"],
                    object=role_stats["object"],
                    aggregate_score=aggregate,
                ),
            )
        )

    ranked.sort(key=lambda item: item.similarity.aggregate_score, reverse=True)
    return ranked[:top_k]


@app.post("/filter_oracol_1", response_model=HypernymExtractionResponse)
async def extract_hypernyms(payload: HypernymExtractionRequest) -> HypernymExtractionResponse:
    """Derive and rank WordNet hypernyms for provided SVO threats."""

    entries = _collect_entries(payload)
    svos = [entry.svo_representation for entry in entries]

    total = len(svos)
    logger.info(
        "Received /filter_oracol_1 request with %d SVO entries (top_k=%d).",
        total,
        payload.top_k,
    )

    if total == 0:
        raise HTTPException(
            status_code=400,
            detail="No SVO representations found. Provide `svos` directly or include entries under `threat`.",
        )

    _ensure_wordnet_resources()
    counters, stats = _aggregate_hypernyms(svos)

    def build_top(counter: collections.Counter) -> List[HypernymRanking]:
        return [
            HypernymRanking(term=term, count=int(count))
            for term, count in counter.most_common(payload.top_k)
        ]

    subject_rankings = build_top(counters["subject"])
    verb_rankings = build_top(counters["verb"])
    object_rankings = build_top(counters["object"])

    hypernym_lists = {
        "subject": [item.term for item in subject_rankings if item.term != NO_MATCH_TOKEN],
        "verb": [item.term for item in verb_rankings if item.term != NO_MATCH_TOKEN],
        "object": [item.term for item in object_rankings if item.term != NO_MATCH_TOKEN],
    }

    ranked_threats = _rank_threats_against_hypernyms(entries, hypernym_lists, payload.top_k)

    metadata = HypernymExtractionMetadata(
        total_svos=total,
        subject_with_match=stats["subject_matches"],
        verb_with_match=stats["verb_matches"],
        object_with_match=stats["object_matches"],
        ranked_threats_considered=len(entries),
    )

    logger.info(
        "Hypernym extraction complete: subject=%d unique, verb=%d, object=%d.",
        len(counters["subject"]),
        len(counters["verb"]),
        len(counters["object"]),
    )

    return HypernymExtractionResponse(
        subject=subject_rankings,
        verb=verb_rankings,
        object=object_rankings,
        ranked_threats=ranked_threats,
        metadata=metadata,
    )


@app.post("/merge_filter_oracol_1", response_model=MergeResponse)
async def merge_filter_oracol_iterations(payload: MergeRequest) -> MergeResponse:
    """Merge multiple /filter_oracol_1 iterations into unified SVO combinations."""

    if not payload.iterations:
        raise HTTPException(status_code=400, detail="Provide at least one iteration to merge.")

    subjects_raw, verbs_raw, objects_raw = _collect_terms_from_iterations(payload)
    subjects_list = sorted(filter(None, subjects_raw))
    verbs_list = sorted(filter(None, verbs_raw))
    objects_list = sorted(filter(None, objects_raw))

    if payload.lemmatize_subjects:
        lemmatized_subjects = _lemmatize_subjects(subjects_list)
        subjects_for_combinations = lemmatized_subjects or subjects_list
        subject_count = len(subjects_for_combinations)
    else:
        subjects_for_combinations = subjects_list
        subject_count = len(subjects_for_combinations)

    combinations = _generate_svo_combinations(
        subjects_for_combinations,
        verbs_list,
        objects_list,
        payload.max_combinations,
    )

    metadata = MergeMetadata(
        total_iterations=len(payload.iterations),
        unique_subjects=subject_count,
        unique_verbs=len(verbs_list),
        unique_objects=len(objects_list),
        generated_combinations=len(combinations),
    )

    return MergeResponse(combinations=combinations, metadata=metadata)

DEFAULT_THRESHOLD = 0.5
RELEVANT_TOPICS = 3


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    """Lazily load and cache the sentence transformer embedder."""

    return SentenceTransformer("stsb-roberta-base-v2")


def _compute_scores(threats: List[str], context_phrase: str) -> Tuple[np.ndarray, List[int], List[float]]:
    """Return relevance scores along with the selected topic ids and similarities."""

    if not threats or not context_phrase:
        logger.info("Skipping scoring because threats or context phrase are missing.")
        return np.zeros(len(threats)), [], []

    if len(threats) == 1:
        logger.info("Only one threat provided; returning zero relevance score to avoid BERTopic transform issues.")
        return np.zeros(len(threats)), [], []

    embedder = _get_embedder()
    model = BERTopic(embedding_model=embedder, nr_topics=None, verbose=False)
    logger.info("Fitting BERTopic model on %d threats.", len(threats))
    model.fit_transform(threats)

    rel_topics, rel_sims = model.find_topics(context_phrase, top_n=RELEVANT_TOPICS)
    if not rel_topics:
        logger.info("No relevant topics found for context phrase.")
        return np.zeros(len(threats)), [], []

    topic_embeddings_all = model.topic_embeddings_
    if topic_embeddings_all is None:
        logger.warning("Topic embeddings not available from BERTopic model.")
        return np.zeros(len(threats)), [], []

    selected_embeddings = []
    valid_topics = []
    valid_sims = []
    for topic, sim in zip(rel_topics, rel_sims):
        vector = topic_embeddings_all[topic]
        if vector is None:
            continue
        selected_embeddings.append(vector)
        valid_topics.append(topic)
        valid_sims.append(sim)

    if not selected_embeddings:
        logger.info("Relevant topics lacked embeddings; returning zero scores.")
        return np.zeros(len(threats)), [], []

    topic_embeddings = np.vstack(selected_embeddings)
    threat_embeddings = embedder.encode(threats, convert_to_numpy=True, show_progress_bar=False)

    similarities = cosine_similarity(threat_embeddings, topic_embeddings)
    scores = similarities.sum(axis=1)
    logger.info("Computed relevance scores for %d threats.", len(scores))

    return scores, valid_topics, valid_sims


@app.post("/filter_oracol_0", response_model=FilterResponse)
async def filter_threats(payload: FilterRequest) -> FilterResponse:
    """Filter the incoming threats using BERTopic relevance against the selected context."""

    logger.info(
        "Received /filter request with %d threats for context '%s'.",
        len(payload.threat),
        payload.context,
    )

    hypernym_phrase = _HYPERNYM_TEMPLATES.get(payload.context, "")
    threat_sentences = [entry.threat for entry in payload.threat]
    scores, rel_topics, rel_sims = _compute_scores(threat_sentences, hypernym_phrase)
    logger.info(
        "Scoring completed. Relevant topics: %s. Mean score: %.3f.",
        rel_topics,
        float(np.mean(scores)) if scores.size else 0.0,
    )

    mean_score = float(np.mean(scores)) if scores.size else 0.0
    filtered_with_scores = [
        (entry, float(score))
        for entry, score in zip(payload.threat, scores)
        if score >= DEFAULT_THRESHOLD
    ]
    filtered_with_scores.sort(key=lambda item: item[1], reverse=True)
    filtered_threats = [entry for entry, _ in filtered_with_scores]
    logger.info(
        "Retained %d/%d threats above threshold %.2f.",
        len(filtered_threats),
        len(payload.threat),
        DEFAULT_THRESHOLD,
    )

    metadata = FilterMetadata(
        threshold=DEFAULT_THRESHOLD,
        total_threats=len(payload.threat),
        retained_threats=len(filtered_threats),
        mean_score=mean_score,
        relevant_topics=[int(topic) for topic in rel_topics],
        relevant_topic_similarities=[float(sim) for sim in rel_sims],
    )

    notes = (
        "Filtered threats using BERTopic relevance scoring."
        if filtered_threats
        else "No threats exceeded the relevance threshold; consider adjusting input or threshold."
    )

    return FilterResponse(
        context=payload.context,
        hypernym_phrase=hypernym_phrase,
        filtered_threats=filtered_threats,
        metadata=metadata,
        notes=notes,
    )




