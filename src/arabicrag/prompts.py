"""
Arabic prompt templates for grounded RAG.

These prompts are tuned for compliance and policy Q&A in Modern Standard
Arabic (MSA). The grounding instruction is critical — without it,
Arabic-capable LLMs frequently invent plausible-sounding regulatory text.
"""

from __future__ import annotations

from .chunking import Chunk

SYSTEM_AR = """\
أنت مساعد متخصص في الإجابة عن الأسئلة المتعلقة باللوائح والأنظمة المالية \
والمصرفية، بناءً على مقاطع نصية مقتبسة من وثائق رسمية فقط.

قواعد صارمة:
1. أجب فقط بناءً على المقاطع المقدمة. لا تستند إلى معرفة عامة.
2. إذا لم تكن المقاطع كافية للإجابة، قل بوضوح: "المعلومات المتاحة لا تكفي للإجابة."
3. اذكر مصدر كل ادعاء بالإشارة إلى رقم المقطع، مثل [مقطع 2].
4. كن دقيقاً ومختصراً. لا تكرر السؤال.
5. استخدم اللغة العربية الفصحى.
"""

USER_TEMPLATE_AR = """\
السؤال:
{question}

المقاطع المرجعية:
{passages}

الإجابة (مع الإشارة إلى أرقام المقاطع):
"""


def format_passages(chunks: list[Chunk]) -> str:
    """Format retrieved chunks for the prompt with stable numbering."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        header = f"[مقطع {i}]"
        if c.section:
            header += f" — {c.section}"
        lines.append(f"{header}\n{c.text}")
    return "\n\n".join(lines)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    """Build OpenAI-compatible chat messages.

    OCI Generative AI's chat API accepts the standard OpenAI message
    format, so this works directly with both the OCI adapter and any
    drop-in replacement model.
    """
    return [
        {"role": "system", "content": SYSTEM_AR},
        {
            "role": "user",
            "content": USER_TEMPLATE_AR.format(
                question=question.strip(),
                passages=format_passages(chunks),
            ),
        },
    ]
