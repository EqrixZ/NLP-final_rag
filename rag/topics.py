"""Knowledge-base topic catalog, templated follow-up suggestions and chemical detection.

Suggestions are generated from document metadata with fixed templates, so the
no-answer card never needs an extra LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import NUM_SUGGESTIONS
from rag.loader import Document

CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "rice": {"th": "ข้าว", "en": "Rice"},
    "durian": {"th": "ทุเรียน", "en": "Durian"},
    "mango": {"th": "มะม่วง", "en": "Mango"},
    "cassava": {"th": "มันสำปะหลัง", "en": "Cassava"},
    "vegetables": {"th": "พืชผัก", "en": "Vegetables"},
    "general": {"th": "ความรู้ทั่วไป", "en": "General"},
}

# Two templates per document kind: [primary, secondary].
_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "disease": {
        "th": ["{t}มีอาการอย่างไร", "วิธีป้องกันและจัดการ{t}"],
        "en": ["What are the symptoms of {t}?", "How can I manage {t}?"],
    },
    "pest": {
        "th": ["{t}ทำลายพืชอย่างไร", "วิธีป้องกันกำจัด{t}"],
        "en": ["What damage does {t} cause?", "How can I control {t}?"],
    },
    "practice": {
        "th": ["หลักสำคัญของ{t}มีอะไรบ้าง", "ขอคำแนะนำเรื่อง{t}"],
        "en": ["What are the key points of {t}?", "Can you give advice on {t}?"],
    },
}


@dataclass(frozen=True)
class Topic:
    """One knowledge-base document as a user-facing topic."""

    source_file: str
    title: str
    topic_th: str
    topic_en: str
    category: str
    kind: str

    def name(self, language: str) -> str:
        return self.topic_th if language == "th" else self.topic_en

    def question(self, language: str, variant: int = 0) -> str:
        """Fill the ``variant``-th template for this topic in ``language``."""
        templates = _TEMPLATES.get(self.kind, _TEMPLATES["practice"])[language]
        topic = self.name(language)
        template = templates[variant % len(templates)]
        # Thai has no word spaces, but "(IPM)มีอะไรบ้าง" reads badly: add a space after ")".
        if language == "th" and topic.endswith(")"):
            template = template.replace("{t}", "{t} ")
        return template.format(t=topic).strip()


def build_topics(documents: list[Document]) -> list[Topic]:
    """Topics in document order (``data/`` is sorted by filename)."""
    return [
        Topic(d.source_file, d.title, d.topic_th, d.topic_en, d.category, d.kind)
        for d in documents
    ]


def main_topics(topics: list[Topic]) -> list[Topic]:
    """First topic of each category — a compact overview of what is covered."""
    seen: set[str] = set()
    result: list[Topic] = []
    for t in topics:
        if t.category not in seen:
            seen.add(t.category)
            result.append(t)
    return result


def make_suggestions(
    language: str,
    closest_files: list[str],
    topics: list[Topic],
    n: int = NUM_SUGGESTIONS,
) -> list[str]:
    """Templated rephrasing suggestions, preferring the topics closest to the query.

    Order: primary template of each closest topic → secondary template of the
    closest topic → alternating templates of the main covered topics.
    """
    by_file = {t.source_file: t for t in topics}
    closest = [by_file[f] for f in dict.fromkeys(closest_files) if f in by_file]
    candidates = [t.question(language, 0) for t in closest]
    if closest:
        candidates.append(closest[0].question(language, 1))
    candidates += [t.question(language, i % 2) for i, t in enumerate(main_topics(topics))]
    return list(dict.fromkeys(candidates))[:n]


# Words that indicate an answer talks about agrochemicals (Thai + English).
_CHEMICAL_PATTERN = re.compile(
    r"สารเคมี|สารป้องกันกำจัด|สารกำจัด|สารฆ่า|ยาฆ่า|สารออกฤทธิ์|สารชะลอการเจริญเติบโต|ฉลาก"
    r"|pesticide|fungicide|insecticide|herbicide|bactericide|nematicide|agrochemical|active ingredient"
    r"|\w*(?:azole|conazole)\b|metalaxyl|mancozeb|carbendazim|paclobutrazol|imidacloprid|pymetrozine|copper"
    r"|โซล|เมทาแลกซิล|แมนโคเซบ|คาร์เบนดาซิม|พาโคลบิวทราโซล|อิมิดาโคลพริด|ไพมีโทรซีน|คอปเปอร์",
    re.IGNORECASE,
)


def mentions_chemicals(text: str) -> bool:
    """True if ``text`` mentions pesticides/agrochemicals (triggers the safety note)."""
    return bool(_CHEMICAL_PATTERN.search(text))
