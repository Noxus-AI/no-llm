from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field


class LmArenaTaskScore(BaseModel):
    elo: float = Field(validation_alias=AliasChoices("rating"))
    final_ranking: int


class LmArenaTaskVariants(BaseModel):
    default: LmArenaTaskScore | None = None
    style_control: LmArenaTaskScore | None = None


class TextLmArenaScore(BaseModel):
    full: LmArenaTaskVariants | None = None
    creative_writing: LmArenaTaskVariants | None = None
    if_task: LmArenaTaskVariants | None = Field(None, validation_alias=AliasChoices("if"))
    math: LmArenaTaskVariants | None = None
    coding: LmArenaTaskVariants | None = None
    hard_6: LmArenaTaskVariants | None = None
    hard_english_6: LmArenaTaskScore | None = None
    multiturn: LmArenaTaskVariants | None = None
    long_user: LmArenaTaskVariants | None = None
    english: LmArenaTaskScore | None = None
    chinese: LmArenaTaskScore | None = None
    french: LmArenaTaskScore | None = None
    german: LmArenaTaskScore | None = None
    spanish: LmArenaTaskScore | None = None
    russian: LmArenaTaskScore | None = None
    japanese: LmArenaTaskScore | None = None
    korean: LmArenaTaskScore | None = None
    no_tie: LmArenaTaskScore | None = None
    no_short: LmArenaTaskScore | None = None
    no_refusal: LmArenaTaskScore | None = None


class VisionLmArenaScore(BaseModel):
    full: LmArenaTaskVariants | None = None
    english: LmArenaTaskVariants | None = None
    chinese: LmArenaTaskVariants | None = None


class BenchmarkScores(BaseModel):
    mt_bench: float | None = None
    mmlu: float | None = Field(None, validation_alias=AliasChoices("mmlu"))
    text_lm_arena: TextLmArenaScore | None = None
    vision_lm_arena: VisionLmArenaScore | None = None

    @classmethod
    def from_lm_arena_json(cls, data: dict) -> BenchmarkScores:
        task_ratings = data.get("task_ratings", {})
        benchmarks = data.get("benchmarks", {})

        text_arena = TextLmArenaScore(
            full=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["full"]),
                style_control=LmArenaTaskScore(**task_ratings["full_style_control"]),
            ),
            creative_writing=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["creative_writing"]),
                style_control=LmArenaTaskScore(**task_ratings["creative_writing_style_control"]),
            ),
            if_task=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["if"]),
                style_control=LmArenaTaskScore(**task_ratings["if_style_control"]),
            ),
            math=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["math"]),
                style_control=LmArenaTaskScore(**task_ratings["math_style_control"]),
            ),
            coding=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["coding"]),
                style_control=LmArenaTaskScore(**task_ratings["coding_style_control"]),
            ),
            hard_6=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["hard_6"]),
                style_control=LmArenaTaskScore(**task_ratings["hard_6_style_control"]),
            ),
            hard_english_6=LmArenaTaskScore(**task_ratings["hard_english_6"]),
            multiturn=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["multiturn"]),
                style_control=LmArenaTaskScore(**task_ratings["multiturn_style_control"]),
            ),
            long_user=LmArenaTaskVariants(
                default=LmArenaTaskScore(**task_ratings["long_user"]),
                style_control=LmArenaTaskScore(**task_ratings["long_user_style_control"]),
            ),
            english=LmArenaTaskScore(**task_ratings["english"]),
            chinese=LmArenaTaskScore(**task_ratings["chinese"]),
            french=LmArenaTaskScore(**task_ratings["french"]),
            german=LmArenaTaskScore(**task_ratings["german"]),
            spanish=LmArenaTaskScore(**task_ratings["spanish"]),
            russian=LmArenaTaskScore(**task_ratings["russian"]),
            japanese=LmArenaTaskScore(**task_ratings["japanese"]),
            korean=LmArenaTaskScore(**task_ratings["korean"]),
            no_tie=LmArenaTaskScore(**task_ratings["no_tie"]),
            no_short=LmArenaTaskScore(**task_ratings["no_short"]),
            no_refusal=LmArenaTaskScore(**task_ratings["no_refusal"]),
        )

        mmlu = float(benchmarks["mmlu"]) if benchmarks.get("mmlu") != "-" else None

        mt_bench = float(benchmarks["mt_bench"]) if benchmarks.get("mt_bench") != "-" else None

        return cls(mt_bench=mt_bench, mmlu=mmlu, text_lm_arena=text_arena)
