from django.contrib import admin

from .models import Answer, Choice, Exam, ExamSession, Question


class ChoiceInline(admin.TabularInline):
    model = Choice
    extra = 4


class QuestionInline(admin.TabularInline):
    model = Question
    extra = 1
    show_change_link = True


@admin.register(Exam)
class ExamAdmin(admin.ModelAdmin):
    list_display = ("title", "duration_minutes", "is_active", "created_by", "created_at")
    inlines = [QuestionInline]


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("exam", "order", "text", "marks")
    list_filter = ("exam",)
    inlines = [ChoiceInline]


@admin.register(ExamSession)
class ExamSessionAdmin(admin.ModelAdmin):
    list_display = (
        "student", "exam", "status", "score", "risk_score",
        "violation_count", "trust_level", "started_at",
    )
    list_filter = ("status", "trust_level", "exam")


admin.site.register(Answer)
