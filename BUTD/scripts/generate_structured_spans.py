import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import spacy


TARGET_DEPS = {"ROOT", "nsubj", "nsubjpass", "dobj", "obj", "attr", "appos", "conj"}
GENERIC_ENTITY_HEADS = {
    "thing", "stuff", "side", "direction", "part", "one", "ones", "group", "set",
    "area", "place", "way", "interest", "hint", "hints", "room", "corner"
}
SCENE_LEVEL_WORDS = {
    "room", "area", "place", "space", "side", "corner", "middle", "center", "end", "front", "back", "wall"
}
SCENE_PART_HEADS = {"wall", "corner", "room", "side", "center", "middle", "end", "front", "back"}
GROUP_HEADS = {"group", "set", "row", "rows", "pair", "pairs", "cluster", "line", "stack"}
DISCOURSE_TOKENS = {
    "ignore", "look", "looking", "see", "choose", "pick", "select", "locate", "find", "please",
    "directly", "straight", "facing", "viewed", "standing", "shown", "talking"
}
COLOR_WORDS = {
    "red", "blue", "green", "white", "black", "brown", "gray", "grey", "yellow", "orange", "purple",
    "pink", "beige", "silver", "gold", "tan", "cream"
}
SIZE_SHAPE_WORDS = {
    "small", "smaller", "smallest", "large", "larger", "largest", "big", "bigger", "biggest",
    "short", "shorter", "shortest", "tall", "taller", "tallest", "long", "longer", "longest",
    "wide", "wider", "widest", "narrow", "narrower", "narrowest", "round", "circular", "square",
    "rectangular", "tiny", "huge", "thin", "thinner", "thinnest", "thick", "thicker", "thickest",
    "slender", "skinny"
}
MATERIAL_STATE_WORDS = {
    "wooden", "metal", "metallic", "plastic", "leather", "glass", "fabric", "open", "closed", "folded",
    "broken", "mounted", "hanging", "striped", "padded", "soft", "hard", "bright", "dark", "light"
}
SPATIAL_ATTR_WORDS = {
    "leftmost", "rightmost", "middle", "center", "centered", "top", "bottom", "upper", "lower",
    "inner", "outer", "left", "right", "front", "back", "rear"
}
VALID_ATTR_WORDS = COLOR_WORDS | SIZE_SHAPE_WORDS | MATERIAL_STATE_WORDS | SPATIAL_ATTR_WORDS
BAD_ATTR_NOUNS = {
    "laptop", "countertop", "desktop", "tabletop", "stovetop", "flower", "flowers", "screen", "keyboard",
    "monitor", "door", "window", "chair", "table", "desk", "cabinet", "toilet", "computer", "wall", "room"
}
PRONOUNS = {"it", "they", "them", "that", "those", "these", "one", "ones"}
TARGET_BLOCKLIST = {
    "the room", "this room", "the living room", "the area", "the place", "the one", "the side", "the corner"
}
RELATION_SPECS = [
    {"surface": "on the left side of", "label": "left of", "max_gap": 18},
    {"surface": "on the right side of", "label": "right of", "max_gap": 18},
    {"surface": "to the left of", "label": "left of", "max_gap": 18},
    {"surface": "to the right of", "label": "right of", "max_gap": 18},
    {"surface": "on the left of", "label": "left of", "max_gap": 18},
    {"surface": "on the right of", "label": "right of", "max_gap": 18},
    {"surface": "left of", "label": "left of", "max_gap": 18},
    {"surface": "right of", "label": "right of", "max_gap": 18},
    {"surface": "in front of", "label": "in front of", "max_gap": 18},
    {"surface": "next to", "label": "next to", "max_gap": 14},
    {"surface": "close to", "label": "near", "max_gap": 14},
    {"surface": "closest to", "label": "closest to", "max_gap": 14},
    {"surface": "nearest to", "label": "nearest to", "max_gap": 14},
    {"surface": "near", "label": "near", "max_gap": 12},
    {"surface": "beside", "label": "beside", "max_gap": 12},
    {"surface": "behind", "label": "behind", "max_gap": 14},
    {"surface": "beneath", "label": "under", "max_gap": 14},
    {"surface": "under", "label": "under", "max_gap": 14},
    {"surface": "below", "label": "below", "max_gap": 14},
    {"surface": "above", "label": "above", "max_gap": 14},
    {"surface": "over", "label": "over", "max_gap": 14},
    {"surface": "between", "label": "between", "max_gap": 18},
    {"surface": "across from", "label": "across from", "max_gap": 18},
    {"surface": "against", "label": "against", "max_gap": 12},
    {"surface": "inside", "label": "inside", "max_gap": 12},
    {"surface": "outside", "label": "outside", "max_gap": 12},
    {"surface": "in the corner of", "label": "in the corner of", "max_gap": 12},
    {"surface": "at the corner of", "label": "at the corner of", "max_gap": 12},
    {"surface": "at the end of", "label": "at the end of", "max_gap": 12},
    {"surface": "on top of", "label": "on top of", "max_gap": 12},
    {"surface": "atop", "label": "on top of", "max_gap": 12},
    {"surface": "by", "label": "by", "max_gap": 6},
    {"surface": "along", "label": "along", "max_gap": 10},
    {"surface": "farthest from", "label": "farthest from", "max_gap": 18},
    {"surface": "furthest from", "label": "furthest from", "max_gap": 18},
    {"surface": "away from", "label": "away from", "max_gap": 18},
    {"surface": "adjacent to", "label": "adjacent to", "max_gap": 12},
]
ALIAS_CLUSTERS = [
    {"sofa", "couch", "loveseat", "love seat"},
    {"chair", "armchair", "arm chair"},
    {"bookshelf", "bookcase"},
    {"trash can", "trashcan", "garbage can", "trash bin", "bin", "storage bin"},
    {"cabinet", "cupboard", "file cabinet", "kitchen cabinet", "kitchen cabinets", "cabinets"},
    {"microwave", "microwave oven"},
    {"nightstand", "night stand"},
    {"telephone", "phone"},
    {"monitor", "screen"},
    {"dresser", "chest", "chest of drawers"},
    {"refrigerator", "fridge", "refridgerator"},
    {"desk", "table", "working table"},
    {"toaster oven", "toaster", "bread toaster"},
    {"bag", "backpack"},
    {"stool", "barstool"},
    {"bathroom stall", "stall", "cubicle"},
    {"tv", "television"},
]
TARGET_TAXONOMY_PREFERRED = {
    "sofa": "couch",
    "couch": "couch",
    "loveseat": "couch",
    "love seat": "couch",
    "chair": "chair",
    "armchair": "chair",
    "arm chair": "chair",
    "office chair": "office chair",
    "bookshelf": "bookshelf",
    "bookcase": "bookshelf",
    "trash can": "trash can",
    "trashcan": "trash can",
    "garbage can": "trash can",
    "trash bin": "trash can",
    "bin": "trash can",
    "storage bin": "storage bin",
    "recycling bin": "recycling bin",
    "cabinet": "cabinet",
    "cupboard": "cabinet",
    "file cabinet": "file cabinet",
    "kitchen cabinet": "kitchen cabinet",
    "kitchen cabinets": "kitchen cabinets",
    "bathroom cabinet": "bathroom cabinet",
    "wardrobe cabinet": "wardrobe cabinet",
    "cabinets": "cabinet",
    "microwave": "microwave",
    "microwave oven": "microwave",
    "nightstand": "nightstand",
    "night stand": "nightstand",
    "telephone": "telephone",
    "phone": "telephone",
    "monitor": "monitor",
    "screen": "monitor",
    "computer screen": "monitor",
    "dresser": "dresser",
    "chest": "dresser",
    "chest of drawers": "dresser",
    "refrigerator": "refrigerator",
    "mini refrigerator": "refrigerator",
    "mini fridge": "mini fridge",
    "fridge": "refrigerator",
    "refridgerator": "refrigerator",
    "desk": "desk",
    "working table": "table",
    "table": "table",
    "end table": "end table",
    "coffee table": "coffee table",
    "dining table": "dining table",
    "round table": "round table",
    "desk lamp": "lamp",
    "toaster oven": "toaster oven",
    "toaster": "toaster",
    "bread toaster": "toaster",
    "bag": "backpack",
    "backpack": "backpack",
    "back pack": "backpack",
    "toilet paper": "toilet paper",
    "toilet paper roll": "toilet paper",
    "paper roll": "toilet paper",
    "toilet paper holder": "toilet paper holder",
    "toilet paper dispenser": "toilet paper dispenser",
    "paper towel dispenser": "paper towel dispenser",
    "stool": "stool",
    "barstool": "stool",
    "bathroom stall": "bathroom stall",
    "bathroom stall door": "bathroom stall door",
    "stall": "bathroom stall",
    "cubicle": "bathroom stall",
    "tv": "tv",
    "television": "tv",
}
COMPOUND_TAXONOMY_PHRASES = {phrase for phrase in TARGET_TAXONOMY_PREFERRED if ' ' in phrase} | {phrase for phrase in TARGET_TAXONOMY_PREFERRED.values() if ' ' in phrase}
OVERGENERIC_TARGET_CANONICALS = {"paper", "table", "chair", "screen", "can", "bin", "lamp", "holder"}
PRIORITY_OVERGENERIC_HEADS = {"can", "chair", "table", "paper", "screen", "lamp", "holder", "bin"}
SPATIAL_ATTR_DIRECTION_MAP = {
    "leftmost": ("left",),
    "left": ("left",),
    "rightmost": ("right",),
    "right": ("right",),
    "front": ("front",),
    "rear": ("back",),
    "back": ("back",),
    "middle": ("middle",),
    "center": ("center",),
    "centered": ("center",),
    "upper": ("upper",),
    "top": ("top",),
    "lower": ("lower",),
    "bottom": ("bottom",),
}
SPATIAL_ATTR_DIMENSION_ORDER = ["upper", "lower", "top", "bottom", "left", "right", "front", "back", "middle", "center"]
CLASS_ALIASES = {
    "office chair": {"office chair", "chair"},
    "sofa chair": {"sofa chair", "chair", "arm chair", "armchair"},
    "books": {"books", "book"},
    "whiteboard": {"whiteboard", "board"},
    "wardrobe closet": {"wardrobe closet", "closet", "wardrobe"},
}
VALID_RELATION_LABELS = {spec["label"] for spec in RELATION_SPECS}
SPATIAL_REL_TO_ATTR = {
    "left of": "left",
    "right of": "right",
    "in front of": "front",
    "behind": "back",
    "at the end of": "end",
}


def normalize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    return str(label).replace("_", " ").strip().lower()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def simple_variants(term: str) -> List[str]:
    out = {term}
    if term.endswith("s"):
        out.add(term[:-1])
    else:
        out.add(term + "s")
    if term.endswith("ies"):
        out.add(term[:-3] + "y")
    if term.endswith("y"):
        out.add(term[:-1] + "ies")
    return sorted(x for x in out if x)


def build_alias_lookup() -> Dict[str, str]:
    lookup = {}
    for cluster in ALIAS_CLUSTERS:
        canonical = sorted(cluster, key=len)[0]
        for item in cluster:
            for variant in simple_variants(item):
                lookup[variant] = canonical
    for key, vals in CLASS_ALIASES.items():
        canonical = lookup.get(key, key)
        for item in {key, *vals}:
            for variant in simple_variants(item):
                lookup.setdefault(variant, canonical)
    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def build_taxonomy_lookup() -> Dict[str, str]:
    lookup = {}
    for item, canonical in TARGET_TAXONOMY_PREFERRED.items():
        for variant in simple_variants(item):
            lookup[variant] = canonical
    for key, vals in CLASS_ALIASES.items():
        canonical = lookup.get(key, TARGET_TAXONOMY_PREFERRED.get(key, key))
        for item in {key, *vals}:
            for variant in simple_variants(item):
                lookup.setdefault(variant, canonical)
    return lookup


TAXONOMY_LOOKUP = build_taxonomy_lookup()


def canonicalize_term(term: str) -> str:
    norm = normalize_text(term)
    return ALIAS_LOOKUP.get(norm, norm)


def taxonomy_canonicalize_term(term: str) -> str:
    norm = normalize_text(term)
    return TAXONOMY_LOOKUP.get(norm, norm)


def strip_leading_function_words(text: str) -> str:
    words = re.findall(r"[A-Za-z]+", normalize_text(text))
    drop = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "their"}
    while words and words[0] in drop:
        words.pop(0)
    return " ".join(words)


def label_aliases(label: Optional[str]) -> List[str]:
    norm = normalize_label(label)
    if not norm:
        return []
    aliases = set(simple_variants(norm))
    aliases.update(CLASS_ALIASES.get(norm, set()))
    canonical = canonicalize_term(norm)
    for key, value in ALIAS_LOOKUP.items():
        if value == canonical:
            aliases.add(key)
    if " " in norm:
        aliases.add(norm.split()[-1])
    expanded = set()
    for alias in aliases:
        expanded.update(simple_variants(alias))
    return sorted(expanded, key=len, reverse=True)


class LayeredSpanParser:
    def __init__(self, model_name: str = "en_core_web_trf"):
        spacy.prefer_gpu()
        self.nlp = spacy.load(model_name)
        self.relation_specs = sorted(RELATION_SPECS, key=lambda x: len(x["surface"]), reverse=True)

    @staticmethod
    def _clean_span(doc, start_tok: int, end_tok: int) -> Optional[Tuple[int, int, str, int, int]]:
        while start_tok < end_tok and doc[start_tok].is_space:
            start_tok += 1
        while end_tok > start_tok and doc[end_tok - 1].is_space:
            end_tok -= 1
        while start_tok < end_tok and doc[start_tok].is_punct:
            start_tok += 1
        while end_tok > start_tok and doc[end_tok - 1].is_punct:
            end_tok -= 1
        if start_tok >= end_tok:
            return None
        span = doc[start_tok:end_tok]
        text = span.text.strip()
        if not text:
            return None
        return span.start_char, span.end_char, text, start_tok, end_tok

    def _make_span_dict(self, doc, start_tok: int, end_tok: int, root_i: int, source: str, extra: Optional[Dict] = None) -> Optional[Dict]:
        cleaned = self._clean_span(doc, start_tok, end_tok)
        if cleaned is None:
            return None
        start, end, text, tok_start, tok_end = cleaned
        item = {
            "text": text,
            "start": start,
            "end": end,
            "token_start": tok_start,
            "token_end": tok_end,
            "root_i": root_i,
            "source": source,
        }
        if extra:
            item.update(extra)
        return item

    @staticmethod
    def _word_set(text: str) -> set:
        return set(re.findall(r"[A-Za-z]+", text.lower()))

    @staticmethod
    def _head_word(text: str) -> str:
        words = re.findall(r"[A-Za-z]+", text.lower())
        return words[-1] if words else ""

    def _canonical_head(self, text: str) -> str:
        head = self._head_word(text)
        return canonicalize_term(head) if head else ""

    def _canonical_surface(self, text: str) -> str:
        return canonicalize_term(text)

    def _label_canonical_forms(self, label: str) -> set:
        return {canonicalize_term(alias) for alias in label_aliases(label)} | {canonicalize_term(label)}

    def _taxonomy_label_forms(self, label: str) -> set:
        forms = {normalize_label(alias) for alias in label_aliases(label)}
        forms.add(normalize_label(label))
        return {taxonomy_canonicalize_term(form) for form in forms if form}

    @staticmethod
    def _approx_match_word(left: str, right: str) -> bool:
        if left == right:
            return True
        if min(len(left), len(right)) < 5:
            return False
        return SequenceMatcher(None, left, right).ratio() >= 0.84

    def _local_phrase_windows(self, doc, target: Dict, radius: int = 3, max_len: int = 4) -> List[Tuple[str, ...]]:
        token_start = target.get("token_start", -1)
        token_end = target.get("token_end", -1)
        if token_start < 0 or token_end <= token_start:
            return []
        start = max(0, token_start - radius)
        end = min(len(doc), token_end + radius)
        words = [normalize_text(tok.text) for tok in doc[start:end] if re.search(r"[A-Za-z]", tok.text)]
        windows = []
        for size in range(2, min(max_len, len(words)) + 1):
            for idx in range(0, len(words) - size + 1):
                windows.append(tuple(words[idx:idx + size]))
        return windows

    def _best_compound_context_match(self, doc, target: Dict, target_label: str) -> str:
        label_norm = normalize_label(target_label)
        label_forms = {normalize_label(alias) for alias in label_aliases(label_norm)} | {label_norm}
        compound_forms = [form for form in label_forms if " " in form]
        if not compound_forms:
            return ""
        windows = self._local_phrase_windows(doc, target)
        if not windows:
            return ""
        best = None
        for form in compound_forms:
            form_words = tuple(re.findall(r"[A-Za-z]+", form))
            if not form_words:
                continue
            for window in windows:
                if len(window) != len(form_words):
                    continue
                if all(self._approx_match_word(w, f) for w, f in zip(window, form_words)):
                    canonical = taxonomy_canonicalize_term(form)
                    score = (
                        canonical == taxonomy_canonicalize_term(label_norm),
                        form == label_norm,
                        len(form_words),
                    )
                    candidate = (score, canonical)
                    if best is None or candidate > best:
                        best = candidate
        return best[1] if best is not None else ""

    def _target_taxonomy_canonical_base(self, doc, target: Dict, target_label: str) -> Tuple[str, str, int, int, int, int, int]:
        raw = strip_leading_function_words(target["text"])
        raw = normalize_text(raw or target["text"])
        label_norm = normalize_label(target_label)
        label_canonical = taxonomy_canonicalize_term(label_norm) if label_norm else ""
        if raw in TAXONOMY_LOOKUP:
            canonical = taxonomy_canonicalize_term(raw)
        else:
            words = re.findall(r"[A-Za-z]+", raw)
            matched = None
            for size in range(min(4, len(words)), 0, -1):
                for start in range(0, len(words) - size + 1):
                    phrase = " ".join(words[start:start + size])
                    if phrase in TAXONOMY_LOOKUP:
                        matched = taxonomy_canonicalize_term(phrase)
                        break
                if matched:
                    break
            if matched:
                canonical = matched
            elif words:
                canonical = taxonomy_canonicalize_term(words[-1])
                if canonical == words[-1] and len(words) > 1:
                    canonical = raw
            else:
                canonical = raw
        compound_context = self._best_compound_context_match(doc, target, label_norm)
        if compound_context and (canonical in OVERGENERIC_TARGET_CANONICALS or canonical == raw or (label_canonical and canonical != label_canonical)):
            canonical = compound_context
        head = canonical if canonical in TAXONOMY_LOOKUP.values() else taxonomy_canonicalize_term(canonical.split()[-1])
        taxonomy_forms = self._taxonomy_label_forms(label_norm)
        alias_matched = int(bool(label_norm) and canonical in taxonomy_forms)
        taxonomy_aligned = int(bool(label_canonical) and canonical == label_canonical)
        target_overgeneric = int(canonical in OVERGENERIC_TARGET_CANONICALS and ((" " in label_norm) or (" " in raw) or not taxonomy_aligned))
        compound_fixed = int(bool(compound_context) and canonical == compound_context and " " in canonical and not target_overgeneric)
        compound_regression_risk = int(target_overgeneric)
        return canonical, head, alias_matched, taxonomy_aligned, compound_fixed, compound_regression_risk, target_overgeneric

    def _specificity_upgrade_candidate(self, doc, ent: Dict, target_label: str) -> Optional[Tuple[Tuple, Dict]]:
        canonical, head, alias_matched, taxonomy_aligned, _, _, target_overgeneric = self._target_taxonomy_canonical_base(doc, ent, target_label)
        if canonical in PRIORITY_OVERGENERIC_HEADS or target_overgeneric:
            return None
        text_norm = normalize_text(ent.get("text", ""))
        score = (
            int(taxonomy_aligned),
            int(alias_matched),
            int(ent.get("is_primary_variant", 0)),
            int(" " in canonical),
            len(re.findall(r"[A-Za-z]+", canonical)),
            round(self._alias_match_score(ent, target_label), 4),
            -ent.get("start", 0),
        )
        return score, {
            "canonical": canonical,
            "head": head,
            "alias_matched": alias_matched,
            "taxonomy_aligned": taxonomy_aligned,
            "target_overgeneric": 0,
            "text": ent.get("text", ""),
        }

    def _repair_overgeneric_target(self, doc, entities: List[Dict], target: Dict, target_label: str, current: Tuple[str, str, int, int, int, int, int]) -> Tuple[str, str, int, int, int, int, int, int, int, int]:
        canonical, head, alias_matched, taxonomy_aligned, compound_fixed, compound_regression_risk, target_overgeneric = current
        overgeneric_target_repaired = 0
        target_specificity_upgraded = 0
        overgeneric_target_remaining = int(target_overgeneric)
        current_head = taxonomy_canonicalize_term(head)
        if current_head not in PRIORITY_OVERGENERIC_HEADS:
            return canonical, head, alias_matched, taxonomy_aligned, compound_fixed, compound_regression_risk, target_overgeneric, overgeneric_target_repaired, overgeneric_target_remaining, target_specificity_upgraded
        target_group = target.get("group_id", -1)
        if target_group < 0:
            return canonical, head, alias_matched, taxonomy_aligned, compound_fixed, compound_regression_risk, target_overgeneric, overgeneric_target_repaired, overgeneric_target_remaining, target_specificity_upgraded
        candidates = []
        for ent in entities:
            if ent.get("group_id", -2) != target_group:
                continue
            cand = self._specificity_upgrade_candidate(doc, ent, target_label)
            if cand is None:
                continue
            candidates.append(cand)
        if not candidates:
            return canonical, head, alias_matched, taxonomy_aligned, compound_fixed, compound_regression_risk, target_overgeneric, overgeneric_target_repaired, overgeneric_target_remaining, target_specificity_upgraded
        best_score, best = max(candidates, key=lambda x: x[0])
        safe_upgrade = bool(best_score[0] or (best_score[1] and best_score[2] and best_score[3]))
        if not safe_upgrade:
            return canonical, head, alias_matched, taxonomy_aligned, compound_fixed, compound_regression_risk, target_overgeneric, overgeneric_target_repaired, overgeneric_target_remaining, target_specificity_upgraded
        canonical = best["canonical"]
        head = best["head"]
        alias_matched = best["alias_matched"]
        taxonomy_aligned = best["taxonomy_aligned"]
        target_overgeneric = 0
        compound_regression_risk = 0
        overgeneric_target_repaired = 1
        overgeneric_target_remaining = 0
        target_specificity_upgraded = 1
        return canonical, head, alias_matched, taxonomy_aligned, compound_fixed, compound_regression_risk, target_overgeneric, overgeneric_target_repaired, overgeneric_target_remaining, target_specificity_upgraded

    def _target_taxonomy_canonical(self, doc, entities: List[Dict], target: Dict, target_label: str) -> Tuple[str, str, int, int, int, int, int, int, int, int]:
        current = self._target_taxonomy_canonical_base(doc, target, target_label)
        return self._repair_overgeneric_target(doc, entities, target, target_label, current)

    def _canonical_spatial_signature(self, text: str) -> Tuple[str, ...]:
        dims = []
        words = re.findall(r"[A-Za-z]+", normalize_text(text))
        for word in words:
            dims.extend(SPATIAL_ATTR_DIRECTION_MAP.get(word, tuple()))
        if not dims:
            return tuple()
        ordered = [dim for dim in SPATIAL_ATTR_DIMENSION_ORDER if dim in dims]
        return tuple(ordered)

    def _dedup_spatial_attributes(self, attrs: List[Dict], target_text: str) -> Tuple[List[Dict], int, int]:
        grouped = {}
        output = []
        dedup_count = 0
        unique_count = 0
        for attr in attrs:
            if attr.get("type") != "spatial_attribute":
                output.append(attr)
                continue
            signature = self._canonical_spatial_signature(attr.get("text", ""))
            if not signature:
                output.append(attr)
                continue
            canonical_text = " ".join(signature)
            key = (normalize_text(attr.get("head") or target_text), signature)
            item = dict(attr)
            item["text"] = canonical_text
            item["canonical_text"] = canonical_text
            prev = grouped.get(key)
            if prev is None:
                grouped[key] = item
                output.append(item)
                unique_count += 1
                continue
            dedup_count += 1
            prev_len = len(re.findall(r"[A-Za-z]+", prev["text"]))
            curr_len = len(re.findall(r"[A-Za-z]+", item["text"]))
            if (curr_len, item["start"], item["end"]) < (prev_len, prev["start"], prev["end"]):
                prev.update(item)
        output.sort(key=lambda x: (x["start"], x["end"], x.get("text", "")))
        return output, dedup_count, unique_count

    def _is_scene_np(self, text: str) -> bool:
        norm = normalize_text(text)
        if norm in TARGET_BLOCKLIST:
            return True
        head = self._head_word(norm)
        return head in SCENE_LEVEL_WORDS or head in GENERIC_ENTITY_HEADS

    def _is_scene_part(self, text: str) -> bool:
        return self._head_word(text) in SCENE_PART_HEADS

    def _is_generic_np(self, text: str) -> bool:
        norm = normalize_text(text)
        words = re.findall(r"[A-Za-z]+", norm)
        if not words:
            return True
        if norm in TARGET_BLOCKLIST:
            return True
        if words[-1] in GENERIC_ENTITY_HEADS:
            return True
        if len(words) == 1 and words[0] in DISCOURSE_TOKENS:
            return True
        return False

    def _entity_modifier_lemmas(self, doc, ent: Dict) -> Tuple[str, ...]:
        if ent.get("token_start", -1) < 0 or ent.get("token_end", -1) <= ent.get("token_start", -1):
            return tuple()
        root_i = ent.get("root_i", -1)
        modifiers = []
        for tok in doc[ent["token_start"]:ent["token_end"]]:
            lemma = tok.lemma_.lower()
            if tok.i == root_i:
                continue
            if tok.dep_ in {"det", "punct", "case", "cc", "conj", "prep"}:
                continue
            if tok.pos_ not in {"ADJ", "NOUN", "PROPN", "NUM"}:
                continue
            if lemma in DISCOURSE_TOKENS or lemma in SCENE_LEVEL_WORDS:
                continue
            modifiers.append(canonicalize_term(lemma))
        return tuple(sorted(set(modifiers)))

    def _entity_meta(self, doc, ent: Dict) -> Dict:
        text = ent["text"]
        head_lemma = self._canonical_head(text)
        modifiers = self._entity_modifier_lemmas(doc, ent)
        if modifiers:
            canonical_text = " ".join(list(modifiers) + [head_lemma])
        else:
            canonical_text = head_lemma or self._canonical_surface(text)
        return {
            "canonical_text": canonical_text,
            "head_lemma": head_lemma,
            "modifier_lemmas": modifiers,
        }

    def _is_object_like(self, doc, ent: Dict) -> bool:
        root = doc[ent["root_i"]] if ent.get("root_i", -1) >= 0 else None
        text = ent["text"]
        words = re.findall(r"[A-Za-z]+", text.lower())
        if not words:
            return False
        if normalize_text(text) in PRONOUNS or self._is_scene_np(text):
            return False
        if root is not None and root.pos_ == "PRON":
            return False
        if any(w in DISCOURSE_TOKENS for w in words):
            return False
        return True

    def _extract_group_object(self, doc, chunk) -> Optional[Dict]:
        if " of " not in chunk.text.lower() or chunk.root.lemma_.lower() not in GROUP_HEADS:
            return None
        for tok in doc[chunk.start:chunk.end]:
            if tok.lemma_.lower() == "of":
                start_tok = tok.i + 1
                if start_tok < chunk.end:
                    return self._make_span_dict(doc, start_tok, chunk.end, doc[chunk.end - 1].i, "group_object", {"group_flag": True})
        return None

    def _chunk_candidates(self, doc) -> List[Dict]:
        candidates = []
        for chunk in doc.noun_chunks:
            root = chunk.root
            text_norm = normalize_text(chunk.text)
            if root.pos_ == "PRON" or text_norm in PRONOUNS:
                continue
            ent = self._make_span_dict(doc, chunk.start, chunk.end, root.i, "noun_chunk")
            if ent is not None:
                ent["group_flag"] = root.lemma_.lower() in GROUP_HEADS
                candidates.append(ent)
            group_obj = self._extract_group_object(doc, chunk)
            if group_obj is not None:
                candidates.append(group_obj)
        return candidates

    def _label_mentions(self, doc, target_label: str) -> List[Dict]:
        mentions = []
        if not target_label:
            return mentions
        lower = doc.text.lower()
        for alias in label_aliases(target_label):
            pattern = re.compile(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])")
            for match in pattern.finditer(lower):
                span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
                if span is None:
                    continue
                mentions.append(self._make_span_dict(doc, span.start, span.end, span.root.i, "label_match"))
                for chunk in doc.noun_chunks:
                    if chunk.start <= span.start and chunk.end >= span.end:
                        mentions.append(self._make_span_dict(doc, chunk.start, chunk.end, chunk.root.i, "label_chunk"))
                        break
        return [m for m in mentions if m is not None]

    def _merge_entities(self, entities: Sequence[Dict]) -> List[Dict]:
        merged = {}
        for ent in entities:
            if ent is None:
                continue
            key = (ent["start"], ent["end"])
            prev = merged.get(key)
            if prev is None or (ent["source"].startswith("label") and not prev["source"].startswith("label")):
                merged[key] = ent
        return sorted(merged.values(), key=lambda x: (x["start"], x["end"]))

    def _entities_compatible(self, left: Dict, right: Dict) -> bool:
        if left["head_lemma"] != right["head_lemma"]:
            return False
        lmods = set(left["modifier_lemmas"])
        rmods = set(right["modifier_lemmas"])
        if not lmods or not rmods:
            return True
        return lmods <= rmods or rmods <= lmods

    def _overlap_or_contain(self, left: Dict, right: Dict) -> bool:
        return not (left["end"] <= right["start"] or right["end"] <= left["start"])

    def _group_entities(self, doc, entities: List[Dict]) -> List[Dict]:
        groups = []
        for ent in entities:
            ent.update(self._entity_meta(doc, ent))
            ent["group_flag"] = ent.get("group_flag", False)
            assigned = None
            for gid, group in enumerate(groups):
                if self._entities_compatible(ent, group[0]) and self._overlap_or_contain(ent, group[0]):
                    assigned = gid
                    break
            if assigned is None:
                groups.append([ent])
                ent["group_id"] = len(groups) - 1
            else:
                groups[assigned].append(ent)
                ent["group_id"] = assigned
        for group in groups:
            primary = max(
                group,
                key=lambda e: (
                    e["source"] == "label_chunk",
                    e["source"] == "noun_chunk",
                    len(e["modifier_lemmas"]),
                    e["end"] - e["start"],
                ),
            )
            for ent in group:
                ent["is_primary_variant"] = int(ent is primary)
        return sorted(entities, key=lambda x: (x["start"], x["end"]))

    def _relation_mentions(self, doc) -> List[Dict]:
        mentions = []
        lower = doc.text.lower()
        for spec in self.relation_specs:
            pattern = re.compile(r"(?<![a-z])" + re.escape(spec["surface"]) + r"(?![a-z])")
            for match in pattern.finditer(lower):
                span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
                if span is None:
                    continue
                mentions.append({
                    "surface": spec["surface"],
                    "label": spec["label"],
                    "start": span.start_char,
                    "end": span.end_char,
                    "token_start": span.start,
                    "token_end": span.end,
                    "text": span.text,
                    "max_gap": spec["max_gap"],
                })
        mentions.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        filtered = []
        used = []
        for mention in mentions:
            if any(not (mention["end"] <= s or mention["start"] >= e) for s, e in used):
                continue
            used.append((mention["start"], mention["end"]))
            filtered.append(mention)
        return filtered

    def _clause_end_token(self, doc, start_tok: int) -> int:
        for tok in doc[start_tok:]:
            if tok.text in {".", ";", ":", ","}:
                return tok.i
        return len(doc)

    def _next_relation_start(self, doc, relation: Dict) -> int:
        for other in self._relation_mentions(doc):
            if other["start"] > relation["start"]:
                return other["token_start"]
        return len(doc)

    def _scene_anchor_chunks(self, doc, relation: Dict) -> List[Dict]:
        clause_end = min(self._clause_end_token(doc, relation["token_end"]), self._next_relation_start(doc, relation))
        out = []
        for chunk in doc.noun_chunks:
            if relation["token_end"] <= chunk.start < clause_end and self._is_scene_part(chunk.text):
                item = self._make_span_dict(doc, chunk.start, chunk.end, chunk.root.i, "scene_anchor")
                if item is not None:
                    item.update(self._entity_meta(doc, item))
                    out.append(item)
        return out

    def _relation_anchor_penalty(self, ent: Dict, relation_mentions: List[Dict]) -> float:
        penalty = 0.0
        for rel in relation_mentions:
            if ent["start"] >= rel["end"] and ent["start"] - rel["end"] <= 28:
                penalty += 1.5
        return penalty

    def _alias_match_score(self, ent: Dict, target_label: str) -> float:
        if not target_label:
            return 0.0
        target_forms = self._label_canonical_forms(target_label)
        text_norm = normalize_text(ent["text"])
        score = 0.0
        if self._canonical_surface(text_norm) in target_forms:
            score = max(score, 6.0)
        if self._canonical_head(text_norm) in target_forms:
            score = max(score, 5.0)
        if self._word_set(text_norm) & self._word_set(target_label):
            score = max(score, 2.5)
        return score

    def _target_score(self, doc, ent: Dict, target_label: str, relation_mentions: List[Dict]) -> float:
        root = doc[ent["root_i"]] if ent.get("root_i", -1) >= 0 else None
        text_norm = normalize_text(ent["text"])
        score = self._alias_match_score(ent, target_label)
        if ent["source"] in {"label_match", "label_chunk"}:
            score += 1.5
        if self._is_object_like(doc, ent):
            score += 1.0
        if root is not None and root.dep_ in TARGET_DEPS:
            score += 2.0
        if root is not None and root.head == root:
            score += 0.6
        if ent.get("group_flag"):
            score -= 0.8
        if not ent.get("is_primary_variant", 1):
            score -= 1.0
        if self._is_scene_np(text_norm):
            score -= 5.0
        if self._is_generic_np(text_norm):
            score -= 3.5
        if len(re.findall(r"[A-Za-z]+", text_norm)) > 8:
            score -= 1.5
        score -= self._relation_anchor_penalty(ent, relation_mentions)
        score -= ent["start"] / 5000.0
        return score

    def choose_target(self, doc, candidates: List[Dict], target_label: str) -> Tuple[Dict, str]:
        relation_mentions = self._relation_mentions(doc)
        primary_candidates = [ent for ent in candidates if ent.get("is_primary_variant", 1)]
        object_candidates = [ent for ent in primary_candidates if self._is_object_like(doc, ent)]
        ranked = object_candidates or primary_candidates or candidates
        if ranked:
            scored = [(self._target_score(doc, ent, target_label, relation_mentions), ent) for ent in ranked]
            best_score, best = max(scored, key=lambda x: x[0])
            if best_score >= 2.0 or not target_label:
                return best, "ranked_entity"
        label = normalize_label(target_label)
        if label:
            first_alpha = next((tok for tok in doc if re.search(r"[A-Za-z]", tok.text)), None)
            start = first_alpha.idx if first_alpha is not None else 0
            target = {
                "text": label,
                "start": start,
                "end": start + len(label),
                "token_start": -1,
                "token_end": -1,
                "root_i": first_alpha.i if first_alpha is not None else 0,
                "source": "constructed_target",
                "group_flag": False,
                "group_id": 0,
                "is_primary_variant": 1,
                "canonical_text": canonicalize_term(label),
                "head_lemma": canonicalize_term(label.split()[-1]),
                "modifier_lemmas": tuple(),
            }
            return target, "constructed_from_label"
        target = {
            "text": doc.text.strip()[:32],
            "start": 0,
            "end": min(len(doc.text), 32),
            "token_start": -1,
            "token_end": -1,
            "root_i": 0,
            "source": "global_fallback",
            "group_flag": False,
            "group_id": 0,
            "is_primary_variant": 1,
            "canonical_text": "",
            "head_lemma": "",
            "modifier_lemmas": tuple(),
        }
        return target, "global_fallback"

    def build_entities(self, doc, target_label: str) -> Tuple[List[Dict], Dict, str, Dict]:
        merged = self._merge_entities(self._label_mentions(doc, target_label) + self._chunk_candidates(doc))
        grouped = self._group_entities(doc, merged)
        target, target_source = self.choose_target(doc, grouped, target_label)
        entities = [target]
        seen = {(target["start"], target["end"]) }
        for ent in grouped:
            key = (ent["start"], ent["end"])
            if key in seen:
                continue
            if self._is_object_like(doc, ent):
                entities.append(ent)
                seen.add(key)
        group_ids = {ent.get("group_id", idx) for idx, ent in enumerate(entities)}
        dedup_meta = {
            "entity_dedup_groups": len(group_ids),
            "entity_redundant_mentions": sum(1 for ent in entities if not ent.get("is_primary_variant", 1)),
        }
        return entities, target, target_source, dedup_meta

    def _recover_pronoun_anchor(self, doc, entities: List[Dict], relation: Dict, target: Dict) -> Optional[Tuple[Dict, int]]:
        target_head = target.get("head_lemma") or self._canonical_head(target["text"])
        target_group = target.get("group_id", -1)
        candidates = []
        for idx, ent in enumerate(entities):
            if ent["start"] == target["start"] and ent["end"] == target["end"]:
                continue
            if ent.get("group_id", -2) == target_group or ent.get("head_lemma") == target_head:
                continue
            if not ent.get("is_primary_variant", 1) or not self._is_object_like(doc, ent):
                continue
            if ent["end"] <= relation["start"] and relation["start"] - ent["end"] <= 120:
                candidates.append((relation["start"] - ent["end"], ent, idx))
        if not candidates:
            return None
        _, ent, idx = min(candidates, key=lambda x: x[0])
        return ent, idx

    def _candidate_anchor_entities(self, doc, entities: List[Dict], relation: Dict, target: Dict) -> List[Tuple[float, Dict, int, bool, bool]]:
        clause_end = min(self._clause_end_token(doc, relation["token_end"]), self._next_relation_start(doc, relation))
        candidates = []
        next_lemma = doc[relation["token_end"]].lemma_.lower() if relation["token_end"] < len(doc) else ""
        if next_lemma in PRONOUNS:
            recovered = self._recover_pronoun_anchor(doc, entities, relation, target)
            if recovered is not None:
                ent, idx = recovered
                candidates.append((0.5, ent, idx, False, True))
        for idx, ent in enumerate(entities):
            if ent["start"] == target["start"] and ent["end"] == target["end"]:
                continue
            if not ent.get("is_primary_variant", 1) or not self._is_object_like(doc, ent):
                continue
            if relation["token_end"] <= ent["token_start"] < clause_end:
                gap = ent["token_start"] - relation["token_end"]
                if gap <= relation["max_gap"]:
                    candidates.append((gap, ent, idx, False, False))
            elif ent["token_end"] <= relation["token_start"] < target.get("token_start", 10 ** 9):
                gap = relation["token_start"] - ent["token_end"]
                if gap <= min(relation["max_gap"], 8):
                    candidates.append((gap + 10, ent, idx, False, False))
        base_idx = len(entities)
        for offset, ent in enumerate(self._scene_anchor_chunks(doc, relation)):
            gap = ent["token_start"] - relation["token_end"]
            if 0 <= gap <= relation["max_gap"]:
                candidates.append((gap + 8, ent, base_idx + offset, True, False))
        return sorted(candidates, key=lambda x: x[0])

    def build_relations(self, doc, entities: List[Dict], target: Dict) -> Tuple[List[Dict], List[Dict], List[int], Dict]:
        relations = []
        anchors = []
        anchor_ids = []
        routed_spatial_attrs = []
        diagnostics = {
            "candidate_relations": 0,
            "valid_tuples": 0,
            "invalid_relation_mentions": 0,
            "half_relation_mentions": 0,
            "long_relation_mentions": 0,
            "scene_anchor_flag": 0,
            "pronoun_anchor_recovered": 0,
            "spatial_info_routed_to_attr": 0,
            "spatial_info_routed_to_scene_tuple": 0,
        }
        seen = set()
        for mention in self._relation_mentions(doc):
            diagnostics["candidate_relations"] += 1
            candidates = self._candidate_anchor_entities(doc, entities, mention, target)
            if not candidates:
                diagnostics["invalid_relation_mentions"] += 1
                mapped = SPATIAL_REL_TO_ATTR.get(mention["label"])
                if mapped:
                    routed_spatial_attrs.append({
                        "text": mapped,
                        "start": mention["start"],
                        "end": mention["end"],
                        "head": target["text"],
                        "type": "spatial_attribute",
                    })
                    diagnostics["spatial_info_routed_to_attr"] += 1
                continue
            score, anchor, anchor_idx, scene_anchor_flag, pronoun_recovered = candidates[0]
            if normalize_text(anchor["text"]) == normalize_text(target["text"]):
                diagnostics["invalid_relation_mentions"] += 1
                continue
            key = (mention["start"], mention["end"], anchor["start"], anchor["end"])
            if key in seen:
                continue
            seen.add(key)
            relation_item = {
                "text": mention["label"],
                "surface_text": mention["text"],
                "start": mention["start"],
                "end": mention["end"],
                "head": target["text"],
                "tail": anchor["text"],
                "tail_start": anchor["start"],
                "tail_end": anchor["end"],
                "pattern": mention["surface"],
                "gap_to_anchor": score,
                "scene_anchor_flag": int(scene_anchor_flag),
                "pronoun_anchor_recovered": int(pronoun_recovered),
                "type": "relation",
            }
            relations.append(relation_item)
            anchors.append({
                "text": anchor["text"],
                "start": anchor["start"],
                "end": anchor["end"],
                "type": "anchor",
                "scene_anchor_flag": int(scene_anchor_flag),
                "pronoun_anchor_recovered": int(pronoun_recovered),
                "canonical_text": anchor.get("canonical_text", canonicalize_term(anchor["text"])),
                "head_lemma": anchor.get("head_lemma", self._canonical_head(anchor["text"])),
            })
            anchor_ids.append(anchor_idx)
            diagnostics["scene_anchor_flag"] += int(scene_anchor_flag)
            diagnostics["pronoun_anchor_recovered"] += int(pronoun_recovered)
            diagnostics["spatial_info_routed_to_scene_tuple"] += int(scene_anchor_flag)
        diagnostics["valid_tuples"] = len(relations)
        return relations, anchors, anchor_ids, routed_spatial_attrs, diagnostics

    def _target_context_tokens(self, doc, target: Dict) -> List:
        token_start = target.get("token_start", -1)
        token_end = target.get("token_end", -1)
        if token_start >= 0 and token_end > token_start:
            for chunk in doc.noun_chunks:
                if chunk.start <= token_start and chunk.end >= token_end:
                    return list(doc[chunk.start:chunk.end])
            return list(doc[token_start:token_end])
        return []

    def build_attributes(self, doc, target: Dict, relations: List[Dict], target_label: str, routed_spatial_attrs: List[Dict]) -> Tuple[List[Dict], Dict]:
        attrs = []
        seen = set()
        diagnostics = {
            "attr_candidates": 0,
            "attribute_pollution_count": 0,
            "spatial_attr_count": 0,
            "spatial_attribute_dedup_rows": 0,
            "spatial_attribute_unique_rows": 0,
        }
        target_tokens = self._target_context_tokens(doc, target)
        token_start = target.get("token_start", -1)
        token_end = target.get("token_end", -1)
        for tok in target_tokens:
            lemma = tok.lemma_.lower()
            word = tok.text.lower()
            if tok.pos_ == "ADJ" and (lemma in VALID_ATTR_WORDS or word in VALID_ATTR_WORDS):
                key = (tok.idx, tok.idx + len(tok.text))
                if key in seen:
                    continue
                seen.add(key)
                kind = "spatial_attribute" if lemma in SPATIAL_ATTR_WORDS or word in SPATIAL_ATTR_WORDS else "attribute"
                attrs.append({"text": tok.text, "start": tok.idx, "end": tok.idx + len(tok.text), "head": target["text"], "type": kind})
                diagnostics["spatial_attr_count"] += int(kind == "spatial_attribute")
        for attr in routed_spatial_attrs:
            key = (attr["text"], attr["start"], attr["end"])
            if key in seen:
                continue
            seen.add(key)
            attrs.append(attr)
            diagnostics["spatial_attr_count"] += 1
        if not relations:
            for tok in doc:
                lemma = tok.lemma_.lower()
                word = tok.text.lower()
                if lemma not in SPATIAL_ATTR_WORDS and word not in SPATIAL_ATTR_WORDS:
                    continue
                if tok.pos_ not in {"ADJ", "ADV", "NOUN"}:
                    continue
                if token_start >= 0 and abs(tok.i - token_start) > 5 and abs(tok.i - token_end) > 5:
                    continue
                key = (tok.text.lower(), tok.idx, tok.idx + len(tok.text))
                if key in seen:
                    continue
                seen.add(key)
                attrs.append({"text": tok.text, "start": tok.idx, "end": tok.idx + len(tok.text), "head": target["text"], "type": "spatial_attribute"})
                diagnostics["spatial_attr_count"] += 1
        attrs, dedup_count, unique_count = self._dedup_spatial_attributes(attrs, target["text"])
        diagnostics["spatial_attribute_dedup_rows"] = dedup_count
        diagnostics["spatial_attribute_unique_rows"] = unique_count
        attrs.sort(key=lambda x: (x["start"], x["end"], x.get("text", "")))
        return attrs, diagnostics

    def compute_parse_meta(self, doc, entities: List[Dict], target: Dict, target_source: str, attrs: List[Dict], relations: List[Dict], anchors: List[Dict], rel_diag: Dict, attr_diag: Dict, dedup_meta: Dict, target_label: str) -> Tuple[float, Dict, Dict]:
        generic_entities = sum(1 for ent in entities[1:] if self._is_generic_np(ent["text"]))
        object_anchors = sum(1 for anchor in anchors if not anchor.get("scene_anchor_flag", 0))
        scene_anchors = sum(1 for anchor in anchors if anchor.get("scene_anchor_flag", 0))
        pronoun_recovered = sum(1 for anchor in anchors if anchor.get("pronoun_anchor_recovered", 0))
        target_match_strength = self._alias_match_score(target, target_label)
        target_text_canonical, target_head_lemma, target_alias_matched, target_taxonomy_aligned, compound_canonical_fixed, compound_canonical_regression_risk, target_overgeneric, overgeneric_target_repaired, overgeneric_target_remaining, target_specificity_upgraded = self._target_taxonomy_canonical(doc, entities, target, target_label)
        if overgeneric_target_repaired:
            target_match_strength = max(target_match_strength, 5.25)
        target_match = int(target_taxonomy_aligned or target_match_strength >= 4.0 or target_source == "constructed_from_label")
        unresolved_alias_ambiguity = int(bool(target_label) and target_alias_matched and not target_taxonomy_aligned)
        confidence = 1.0
        confidence -= 0.24 * (1 - min(target_match_strength / 6.0, 1.0))
        confidence -= 0.08 * (1 - target_taxonomy_aligned)
        confidence -= 0.04 * unresolved_alias_ambiguity
        confidence -= 0.04 * target_overgeneric
        confidence -= 0.05 * min(dedup_meta.get("entity_redundant_mentions", 0), 3)
        confidence -= 0.08 * min(generic_entities, 3)
        confidence -= 0.08 * min(attr_diag["attribute_pollution_count"], 2)
        confidence -= 0.06 * min(attr_diag.get("spatial_attribute_dedup_rows", 0), 2)
        confidence -= 0.10 * min(rel_diag["invalid_relation_mentions"], 2)
        confidence -= 0.08 * min(rel_diag["long_relation_mentions"], 2)
        confidence -= 0.08 * min(scene_anchors, 2)
        confidence -= 0.05 * min(pronoun_recovered, 2)
        if rel_diag["spatial_info_routed_to_attr"]:
            confidence -= 0.03 * min(rel_diag["spatial_info_routed_to_attr"], 2)
        confidence = max(0.0, min(1.0, confidence))
        coverage = {
            "has_target": 1,
            "target_source": target_source,
            "target_label_match": target_match,
            "target_match_strength": round(target_match_strength, 4),
            "target_alias_matched": target_alias_matched,
            "target_taxonomy_aligned": target_taxonomy_aligned,
            "compound_canonical_fixed": compound_canonical_fixed,
            "compound_canonical_regression_risk": compound_canonical_regression_risk,
            "target_overgeneric_canonical": target_overgeneric,
            "overgeneric_target_repaired": overgeneric_target_repaired,
            "overgeneric_target_remaining": overgeneric_target_remaining,
            "target_specificity_upgraded": target_specificity_upgraded,
            "unresolved_alias_ambiguity": unresolved_alias_ambiguity,
            "target_text_raw": target["text"],
            "target_text_canonical": target_text_canonical,
            "target_head_lemma": target_head_lemma,
            "num_entities": len(entities),
            "entity_dedup_groups": dedup_meta.get("entity_dedup_groups", len(entities)),
            "entity_redundant_mentions": dedup_meta.get("entity_redundant_mentions", 0),
            "generic_entity_count": generic_entities,
            "num_attrs": len(attrs),
            "attribute_pollution_count": attr_diag["attribute_pollution_count"],
            "spatial_attribute_rows": int(any(attr.get("type") == "spatial_attribute" for attr in attrs)),
            "spatial_attribute_dedup_rows": attr_diag.get("spatial_attribute_dedup_rows", 0),
            "spatial_attribute_unique_rows": attr_diag.get("spatial_attribute_unique_rows", 0),
            "num_relations": len(relations),
            "valid_tuple_count": len(relations),
            "candidate_relation_count": rel_diag["candidate_relations"],
            "invalid_relation_count": rel_diag["invalid_relation_mentions"],
            "half_relation_count": rel_diag["half_relation_mentions"],
            "long_relation_count": rel_diag["long_relation_mentions"],
            "object_like_anchor_count": object_anchors,
            "scene_anchor_flag": scene_anchors,
            "pronoun_anchor_recovered": pronoun_recovered,
            "spatial_info_routed_to_attr": rel_diag["spatial_info_routed_to_attr"],
            "spatial_info_routed_to_scene_tuple": rel_diag["spatial_info_routed_to_scene_tuple"],
            "has_valid_tuple": int(len(relations) > 0),
        }
        slot_mask = {
            "global_slot": 1,
            "target_slot": 1,
            "attr_slot": int(len(attrs) > 0),
            "rel_slots": [1] * len(relations),
            "anchor_slots": [1] * len(anchors),
        }
        return confidence, coverage, slot_mask

    def parse_doc(self, doc, target_label: Optional[str] = None) -> Dict:
        target_label = normalize_label(target_label)
        entities, target, target_source, dedup_meta = self.build_entities(doc, target_label)
        relations, anchors, anchor_ids, routed_spatial_attrs, rel_diag = self.build_relations(doc, entities, target)
        attrs, attr_diag = self.build_attributes(doc, target, relations, target_label, routed_spatial_attrs)
        confidence, coverage, slot_mask = self.compute_parse_meta(doc, entities, target, target_source, attrs, relations, anchors, rel_diag, attr_diag, dedup_meta, target_label)
        entity_records = []
        for ent in entities:
            entity_records.append({
                "text": ent["text"],
                "start": ent["start"],
                "end": ent["end"],
                "group_flag": ent.get("group_flag", False),
                "source": ent.get("source", "entity"),
                "canonical_text": ent.get("canonical_text", canonicalize_term(ent["text"])),
                "head_lemma": ent.get("head_lemma", self._canonical_head(ent["text"])),
                "group_id": ent.get("group_id", 0),
                "is_primary_variant": ent.get("is_primary_variant", 1),
            })
        fixed_anchor_ids = []
        for anchor, anchor_idx in zip(anchors, anchor_ids):
            if 0 <= anchor_idx < len(entity_records):
                fixed_anchor_ids.append(anchor_idx)
                continue
            entity_records.append({
                "text": anchor["text"],
                "start": anchor["start"],
                "end": anchor["end"],
                "group_flag": False,
                "source": "scene_anchor" if anchor.get("scene_anchor_flag", 0) else "recovered_anchor",
                "canonical_text": anchor.get("canonical_text", canonicalize_term(anchor["text"])),
                "head_lemma": anchor.get("head_lemma", self._canonical_head(anchor["text"])),
                "group_id": max([e.get("group_id", 0) for e in entity_records] + [0]) + 1,
                "is_primary_variant": 1,
            })
            fixed_anchor_ids.append(len(entity_records) - 1)
        return {
            "global_slot": {"text": doc.text, "start": 0, "end": len(doc.text), "type": "global"},
            "target_slot": {"text": target["text"], "start": target["start"], "end": target["end"], "type": "target"},
            "attr_slot": {"items": attrs, "is_empty": len(attrs) == 0},
            "rel_slots": relations,
            "anchor_slots": anchors,
            "slot_mask": slot_mask,
            "parse_confidence": confidence,
            "coverage_stats": coverage,
            "entities": entity_records,
            "attributes": attrs,
            "relations": relations,
            "anchor_ids": fixed_anchor_ids,
            "anchors_types": [anchor["text"] for anchor in anchors],
            "target_text": target["text"],
            "target_source": target_source,
            "target_text_raw": target["text"],
            "target_text_canonical": coverage["target_text_canonical"],
            "target_head_lemma": coverage["target_head_lemma"],
            "target_alias_matched": coverage["target_alias_matched"],
            "target_taxonomy_aligned": coverage["target_taxonomy_aligned"],
            "entity_canonical_texts": [e["canonical_text"] for e in entity_records],
            "entity_group_ids": [e["group_id"] for e in entity_records],
            "entity_primary_mask": [e["is_primary_variant"] for e in entity_records],
            "entity_head_lemmas": [e["head_lemma"] for e in entity_records],
        }

    def parse_texts(self, texts: List[str], target_labels: Optional[List[str]] = None, batch_size: int = 128) -> List[Dict]:
        labels = target_labels or [""] * len(texts)
        outputs = []
        for idx, (doc, label) in enumerate(zip(self.nlp.pipe(texts, batch_size=batch_size), labels), start=1):
            outputs.append(self.parse_doc(doc, label))
            if idx % 2000 == 0:
                print(f"parsed {idx}/{len(texts)}")
        return outputs


def process_referit_csv(input_path: Path, output_path: Path, parser: LayeredSpanParser):
    with input_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    extras = [
        "global_slot", "target_slot", "attr_slot", "rel_slots", "anchor_slots", "slot_mask",
        "entities", "attributes", "relations", "anchors_types", "anchor_ids", "target_text",
        "target_source", "parse_confidence", "coverage_stats", "target_text_raw", "target_text_canonical",
        "target_head_lemma", "target_alias_matched", "target_taxonomy_aligned", "entity_canonical_texts", "entity_group_ids",
        "entity_primary_mask", "entity_head_lemmas"
    ]
    for extra in extras:
        if extra not in fieldnames:
            fieldnames.append(extra)
    parsed = parser.parse_texts([row["utterance"] for row in rows], [row.get("instance_type") for row in rows])
    for row, item in zip(rows, parsed):
        for key in [
            "global_slot", "target_slot", "attr_slot", "rel_slots", "anchor_slots", "slot_mask", "entities",
            "attributes", "relations", "anchors_types", "anchor_ids", "coverage_stats", "entity_canonical_texts",
            "entity_group_ids", "entity_primary_mask", "entity_head_lemmas"
        ]:
            row[key] = json.dumps(item[key], ensure_ascii=False)
        for key in ["target_text", "target_source", "parse_confidence", "target_text_raw", "target_text_canonical", "target_head_lemma", "target_alias_matched", "target_taxonomy_aligned"]:
            row[key] = item[key]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_scanrefer_json(input_path: Path, output_path: Path, parser: LayeredSpanParser):
    data = json.load(input_path.open())
    utterances = [" ".join(item["token"]) for item in data]
    labels = [str(item.get("object_name", "")).replace("_", " ") for item in data]
    parsed = parser.parse_texts(utterances, labels)
    out = []
    for item, utt, parsed_item in zip(data, utterances, parsed):
        row = dict(item)
        row["utterance"] = utt
        row.update(parsed_item)
        out.append(row)
    with output_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False)


def _read_referit_rows(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "utterance": row.get("utterance", ""),
                "target_label": row.get("instance_type", ""),
                "target_text": row.get("target_text", ""),
                "target_text_canonical": row.get("target_text_canonical", ""),
                "target_alias_matched": int(row.get("target_alias_matched", 0) or 0),
                "target_taxonomy_aligned": int(row.get("target_taxonomy_aligned", 0) or 0),
                "entities": json.loads(row["entities"]),
                "attributes": json.loads(row["attributes"]),
                "relations": json.loads(row["relations"]),
                "anchors": json.loads(row.get("anchor_slots", "[]")),
                "anchor_ids": json.loads(row["anchor_ids"]),
                "coverage_stats": json.loads(row["coverage_stats"]),
            }


def _read_scanrefer_rows(path: Path) -> Iterable[Dict]:
    for row in json.load(path.open()):
        yield {
            "utterance": row.get("utterance") or " ".join(row.get("token", [])),
            "target_label": row.get("object_name", ""),
            "target_text": row.get("target_text", ""),
            "target_text_canonical": row.get("target_text_canonical", ""),
            "target_alias_matched": int(row.get("target_alias_matched", 0) or 0),
            "target_taxonomy_aligned": int(row.get("target_taxonomy_aligned", 0) or 0),
            "entities": row["entities"],
            "attributes": row["attributes"],
            "relations": row["relations"],
            "anchors": row.get("anchor_slots", []),
            "anchor_ids": row["anchor_ids"],
            "coverage_stats": row.get("coverage_stats", {}),
        }


def validate_rows(rows: Iterable[Dict]) -> Dict:
    total = entity_empty = attr_empty = rel_empty = valid_tuple_empty = 0
    anchor_mismatch = anchor_oob = 0
    scene_anchor_flag = pronoun_anchor_recovered = spatial_attribute_rows = 0
    spatial_attribute_dedup_rows = spatial_attribute_unique_rows = 0
    entity_dedup_groups = target_alias_matched_rows = target_taxonomy_aligned_rows = 0
    compound_canonical_fixed_rows = compound_canonical_regression_risk_rows = target_overgeneric_canonical_rows = 0
    overgeneric_target_repaired_rows = overgeneric_target_remaining_rows = target_specificity_upgraded_rows = 0
    for row in rows:
        cov = row.get("coverage_stats", {})
        total += 1
        entity_empty += int(len(row["entities"]) == 0)
        attr_empty += int(len(row["attributes"]) == 0)
        rel_empty += int(len(row["relations"]) == 0)
        valid_tuple_empty += int(cov.get("valid_tuple_count", 0) == 0)
        anchor_mismatch += int(len(row["anchor_ids"]) != len(row["relations"]))
        anchor_oob += int(any(a < 0 or a >= len(row["entities"]) for a in row["anchor_ids"]))
        scene_anchor_flag += int(cov.get("scene_anchor_flag", 0))
        pronoun_anchor_recovered += int(cov.get("pronoun_anchor_recovered", 0))
        spatial_attribute_rows += int(cov.get("spatial_attribute_rows", 0))
        spatial_attribute_dedup_rows += int(cov.get("spatial_attribute_dedup_rows", 0))
        spatial_attribute_unique_rows += int(cov.get("spatial_attribute_unique_rows", 0))
        entity_dedup_groups += int(max(0, cov.get("num_entities", 0) - cov.get("entity_dedup_groups", 0)))
        target_alias_matched_rows += int(cov.get("target_alias_matched", 0))
        target_taxonomy_aligned_rows += int(cov.get("target_taxonomy_aligned", 0))
        compound_canonical_fixed_rows += int(cov.get("compound_canonical_fixed", 0))
        compound_canonical_regression_risk_rows += int(cov.get("compound_canonical_regression_risk", 0))
        target_overgeneric_canonical_rows += int(cov.get("target_overgeneric_canonical", 0))
        overgeneric_target_repaired_rows += int(cov.get("overgeneric_target_repaired", 0))
        overgeneric_target_remaining_rows += int(cov.get("overgeneric_target_remaining", 0))
        target_specificity_upgraded_rows += int(cov.get("target_specificity_upgraded", 0))
    return {
        "total": total,
        "entity_empty": entity_empty,
        "attr_empty": attr_empty,
        "rel_empty": rel_empty,
        "valid_tuple_empty": valid_tuple_empty,
        "anchor_len_mismatch": anchor_mismatch,
        "anchor_oob": anchor_oob,
        "scene_anchor_flag": scene_anchor_flag,
        "pronoun_anchor_recovered": pronoun_anchor_recovered,
        "spatial_attribute_rows": spatial_attribute_rows,
        "spatial_attribute_dedup_rows": spatial_attribute_dedup_rows,
        "spatial_attribute_unique_rows": spatial_attribute_unique_rows,
        "entity_dedup_groups": entity_dedup_groups,
        "target_alias_matched_rows": target_alias_matched_rows,
        "target_taxonomy_aligned_rows": target_taxonomy_aligned_rows,
        "compound_canonical_fixed_rows": compound_canonical_fixed_rows,
        "compound_canonical_regression_risk_rows": compound_canonical_regression_risk_rows,
        "target_overgeneric_canonical_rows": target_overgeneric_canonical_rows,
        "overgeneric_target_repaired_rows": overgeneric_target_repaired_rows,
        "overgeneric_target_remaining_rows": overgeneric_target_remaining_rows,
        "target_specificity_upgraded_rows": target_specificity_upgraded_rows,
    }


def _is_attr_pollution(attr_text: str, target_label: str) -> bool:
    words = set(re.findall(r"[A-Za-z]+", normalize_text(attr_text)))
    if not words:
        return True
    if words <= COLOR_WORDS or words <= SIZE_SHAPE_WORDS or words <= MATERIAL_STATE_WORDS or words <= SPATIAL_ATTR_WORDS:
        return False
    target_forms = {canonicalize_term(x) for x in re.findall(r"[A-Za-z]+", target_label)}
    return any(canonicalize_term(w) in target_forms for w in words) or any(w in BAD_ATTR_NOUNS or w in SCENE_LEVEL_WORDS for w in words)


def semantic_audit(rows: Iterable[Dict]) -> Dict:
    total = 0
    counts = Counter()
    examples = defaultdict(list)
    for row in rows:
        total += 1
        utter = row["utterance"]
        target_label = normalize_label(row["target_label"])
        target_text = normalize_label(row["target_text"])
        target_canonical = normalize_label(row.get("target_text_canonical", ""))
        entities = row["entities"]
        attrs = row["attributes"]
        rels = row["relations"]
        anchors = row.get("anchors", [])
        coverage = row.get("coverage_stats", {})

        taxonomy_label = taxonomy_canonicalize_term(target_label) if target_label else ""
        target_alias_matched = int(coverage.get("target_alias_matched", 0))
        target_taxonomy_aligned = int(coverage.get("target_taxonomy_aligned", 0))
        target_label_match = int(coverage.get("target_label_match", 0))
        if target_alias_matched:
            counts["target_alias_matched_rows"] += 1
        if target_taxonomy_aligned:
            counts["target_taxonomy_aligned_rows"] += 1
            if len(examples["taxonomy_alignment_corrected"]) < 12 and target_canonical and target_canonical == taxonomy_label and target_text != target_canonical:
                examples["taxonomy_alignment_corrected"].append({
                    "utterance": utter,
                    "target_label": target_label,
                    "target_text": target_text,
                    "target_text_canonical": target_canonical,
                    "target_head_lemma": coverage.get("target_head_lemma", ""),
                })
        if int(coverage.get("compound_canonical_fixed", 0)):
            counts["compound_canonical_fixed_rows"] += 1
            if len(examples["compound_canonical_corrected"]) < 12:
                examples["compound_canonical_corrected"].append({
                    "utterance": utter,
                    "target_label": target_label,
                    "target_text": target_text,
                    "target_text_canonical": target_canonical,
                    "target_head_lemma": coverage.get("target_head_lemma", ""),
                })
        if int(coverage.get("overgeneric_target_repaired", 0)):
            counts["overgeneric_target_repaired_rows"] += 1
            if len(examples["overgeneric_target_repaired"]) < 12:
                examples["overgeneric_target_repaired"].append({
                    "utterance": utter,
                    "target_label": target_label,
                    "target_text": target_text,
                    "target_text_canonical": target_canonical,
                    "target_head_lemma": coverage.get("target_head_lemma", ""),
                })
        if int(coverage.get("overgeneric_target_remaining", 0)):
            counts["overgeneric_target_remaining_rows"] += 1
            if len(examples["overgeneric_target_remaining"]) < 12:
                examples["overgeneric_target_remaining"].append({
                    "utterance": utter,
                    "target_label": target_label,
                    "target_text": target_text,
                    "target_text_canonical": target_canonical,
                })
        if int(coverage.get("target_specificity_upgraded", 0)):
            counts["target_specificity_upgraded_rows"] += 1
        if int(coverage.get("compound_canonical_regression_risk", 0)):
            counts["compound_canonical_regression_risk_rows"] += 1
            if len(examples["compound_canonical_regression_risk"]) < 12:
                examples["compound_canonical_regression_risk"].append({
                    "utterance": utter,
                    "target_label": target_label,
                    "target_text": target_text,
                    "target_text_canonical": target_canonical,
                })
        if int(coverage.get("target_overgeneric_canonical", 0)):
            counts["target_overgeneric_canonical_rows"] += 1
            if len(examples["target_overgeneric_canonical"]) < 12:
                examples["target_overgeneric_canonical"].append({
                    "utterance": utter,
                    "target_label": target_label,
                    "target_text": target_text,
                    "target_text_canonical": target_canonical,
                })
        if not target_label_match:
            counts["target_label_mismatch_rows"] += 1
            if len(examples["target_alias_mismatch"]) < 12:
                examples["target_alias_mismatch"].append({
                    "utterance": utter,
                    "target_label": target_label,
                    "target_text": target_text,
                    "target_text_canonical": target_canonical,
                    "target_head_lemma": coverage.get("target_head_lemma", ""),
                })

        generic_candidates = [ent for ent in entities[1:] if ent.get("source") != "scene_anchor"]
        if any(normalize_text(ent["text"]) in TARGET_BLOCKLIST or normalize_text(ent["text"]).endswith(" room") for ent in generic_candidates):
            counts["generic_entity_rows"] += 1

        if any(_is_attr_pollution(attr["text"], target_label) for attr in attrs):
            counts["attribute_pollution_rows"] += 1
            if len(examples["attribute_pollution_case"]) < 12:
                bad = next(attr["text"] for attr in attrs if _is_attr_pollution(attr["text"], target_label))
                examples["attribute_pollution_case"].append({"utterance": utter, "attribute": bad})

        group_map = defaultdict(list)
        for ent in entities:
            group_map[ent.get("group_id", -1)].append(ent)
        if any(len(v) > 1 for v in group_map.values()):
            counts["entity_dedup_groups"] += sum(1 for v in group_map.values() if len(v) > 1)
            if len(examples["entity_dedup_grouping_corrected"]) < 12:
                groups = [[e["text"] for e in vals] for vals in group_map.values() if len(vals) > 1]
                examples["entity_dedup_grouping_corrected"].append({"utterance": utter, "groups": groups[:3]})

        spatial_attrs = [attr for attr in attrs if attr.get("type") == "spatial_attribute"]
        if spatial_attrs:
            counts["spatial_attribute_rows"] += 1
            if len(examples["spatial_attribute_case"]) < 12:
                examples["spatial_attribute_case"].append({"utterance": utter, "attributes": spatial_attrs})
        if coverage.get("spatial_attribute_dedup_rows", 0):
            counts["spatial_attribute_dedup_rows"] += 1
            if len(examples["spatial_attribute_deduplicated"]) < 12:
                examples["spatial_attribute_deduplicated"].append({
                    "utterance": utter,
                    "attributes": spatial_attrs,
                    "coverage_stats": coverage,
                })

        bad_rel = False
        for rel, anchor in zip(rels, anchors):
            if rel["text"] not in VALID_RELATION_LABELS or rel.get("gap_to_anchor", 0) > 40:
                bad_rel = True
            if anchor.get("scene_anchor_flag", 0) and len(examples["scene_anchor_low_conf_tuple"]) < 12:
                examples["scene_anchor_low_conf_tuple"].append({"utterance": utter, "relation": rel, "anchor": anchor})
            if anchor.get("pronoun_anchor_recovered", 0) and len(examples["pronoun_anchor_recovered_validated"]) < 12:
                examples["pronoun_anchor_recovered_validated"].append({"utterance": utter, "relation": rel, "anchor": anchor})
        if bad_rel:
            counts["bad_relation_rows"] += 1

        if coverage.get("candidate_relation_count", 0) > 0 and coverage.get("valid_tuple_count", 0) == 0:
            counts["spatial_info_without_tuple_rows"] += 1
            if len(examples["intentionally_unresolved_for_quality"]) < 12:
                examples["intentionally_unresolved_for_quality"].append({"utterance": utter, "coverage_stats": coverage})

        if coverage.get("spatial_info_routed_to_attr", 0) > 0 and len(examples["spatial_info_routed_to_attr"]) < 12:
            examples["spatial_info_routed_to_attr"].append({"utterance": utter, "attributes": spatial_attrs, "coverage_stats": coverage})

    return {
        "total": total,
        "entity_empty": None,
        "attr_empty": None,
        "rel_empty": None,
        "valid_tuple_empty": None,
        "target_label_mismatch_rows": counts["target_label_mismatch_rows"],
        "bad_relation_rows": counts["bad_relation_rows"],
        "generic_entity_rows": counts["generic_entity_rows"],
        "attribute_pollution_rows": counts["attribute_pollution_rows"],
        "spatial_info_without_tuple_rows": counts["spatial_info_without_tuple_rows"],
        "scene_anchor_flag": None,
        "pronoun_anchor_recovered": None,
        "spatial_attribute_rows": counts["spatial_attribute_rows"],
        "spatial_attribute_dedup_rows": counts["spatial_attribute_dedup_rows"],
        "entity_dedup_groups": counts["entity_dedup_groups"],
        "target_alias_matched_rows": counts["target_alias_matched_rows"],
        "target_taxonomy_aligned_rows": counts["target_taxonomy_aligned_rows"],
        "compound_canonical_fixed_rows": counts["compound_canonical_fixed_rows"],
        "compound_canonical_regression_risk_rows": counts["compound_canonical_regression_risk_rows"],
        "target_overgeneric_canonical_rows": counts["target_overgeneric_canonical_rows"],
        "overgeneric_target_repaired_rows": counts["overgeneric_target_repaired_rows"],
        "overgeneric_target_remaining_rows": counts["overgeneric_target_remaining_rows"],
        "target_specificity_upgraded_rows": counts["target_specificity_upgraded_rows"],
        "examples": dict(examples),
    }


def build_audit_for_path(path: Path, kind: str) -> Tuple[Dict, Dict]:
    rows = list(_read_referit_rows(path) if kind == "csv" else _read_scanrefer_rows(path))
    stats = validate_rows(rows)
    audit = semantic_audit(rows)
    merged = dict(audit)
    merged.update(stats)
    return stats, merged


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data-root", required=True)
    cli.add_argument("--model", default="en_core_web_trf")
    args = cli.parse_args()

    root = Path(args.data_root)
    parser = LayeredSpanParser(args.model)

    sr3d_out = root / "refer_it_3d" / "sr3d_spacy.csv"
    nr3d_out = root / "refer_it_3d" / "nr3d_spacy.csv"
    scan_train_out = root / "scanrefer" / "ScanRefer_filtered_train_spacy.json"
    scan_val_out = root / "scanrefer" / "ScanRefer_filtered_val_spacy.json"

    process_referit_csv(root / "refer_it_3d" / "sr3d.csv", sr3d_out, parser)
    process_referit_csv(root / "refer_it_3d" / "nr3d.csv", nr3d_out, parser)
    process_scanrefer_json(root / "scanrefer" / "ScanRefer_filtered_train.json", scan_train_out, parser)
    process_scanrefer_json(root / "scanrefer" / "ScanRefer_filtered_val.json", scan_val_out, parser)

    stats = {
        "sr3d_spacy.csv": validate_rows(_read_referit_rows(sr3d_out)),
        "nr3d_spacy.csv": validate_rows(_read_referit_rows(nr3d_out)),
        "ScanRefer_filtered_train_spacy.json": validate_rows(_read_scanrefer_rows(scan_train_out)),
        "ScanRefer_filtered_val_spacy.json": validate_rows(_read_scanrefer_rows(scan_val_out)),
    }
    stats_path = root / "structured_span_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    audit = {
        "sr3d_spacy.csv": build_audit_for_path(sr3d_out, "csv")[1],
        "nr3d_spacy.csv": build_audit_for_path(nr3d_out, "csv")[1],
        "ScanRefer_filtered_train_spacy.json": build_audit_for_path(scan_train_out, "json")[1],
        "ScanRefer_filtered_val_spacy.json": build_audit_for_path(scan_val_out, "json")[1],
    }
    audit_path = root / "structured_span_semantic_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("saved_stats", stats_path)
    print("saved_audit", audit_path)


if __name__ == "__main__":
    main()
