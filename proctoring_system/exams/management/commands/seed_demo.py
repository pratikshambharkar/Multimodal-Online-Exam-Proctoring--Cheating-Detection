"""Seed demo data for RAIM: an instructor, a student and two ready-to-take exams.

Idempotent — safe to run any number of times; existing rows are reused, never
duplicated. Usage: ``python manage.py seed_demo``.
"""
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import transaction

from accounts.models import Profile
from exams.models import Choice, Exam, Question

CS_EXAM = {
    "title": "Computer Science Fundamentals",
    "description": (
        "Core MCQs covering data structures, operating systems, computer "
        "networks and database management systems."
    ),
    "duration_minutes": 20,
    "questions": [
        (
            "Which data structure follows the Last-In-First-Out (LIFO) principle?",
            [("Stack", True), ("Queue", False), ("Linked list", False), ("Binary tree", False)],
        ),
        (
            "What is the worst-case time complexity of searching for a key in a "
            "balanced binary search tree with n nodes?",
            [("O(log n)", True), ("O(1)", False), ("O(n)", False), ("O(n log n)", False)],
        ),
        (
            "Which data structure is used to implement breadth-first search (BFS) "
            "of a graph?",
            [("Queue", True), ("Stack", False), ("Hash table", False), ("Binary heap", False)],
        ),
        (
            "In an operating system, thrashing is primarily caused by:",
            [
                ("Excessive page faults due to insufficient memory frames", True),
                ("Too many CPU registers being used", False),
                ("A very fast secondary storage device", False),
                ("Running only one process at a time", False),
            ],
        ),
        (
            "Which of the following is NOT one of the four necessary conditions "
            "for deadlock?",
            [
                ("Preemption", True),
                ("Mutual exclusion", False),
                ("Hold and wait", False),
                ("Circular wait", False),
            ],
        ),
        (
            "Which OSI model layer is responsible for reliable end-to-end delivery "
            "of data between hosts?",
            [
                ("Transport layer", True),
                ("Network layer", False),
                ("Data link layer", False),
                ("Session layer", False),
            ],
        ),
        (
            "Which protocol resolves an IP address into the corresponding MAC "
            "address on a local network?",
            [("ARP", True), ("DNS", False), ("DHCP", False), ("ICMP", False)],
        ),
        (
            "A relation that is in 2NF and has no transitive dependency of "
            "non-key attributes on the primary key is in:",
            [
                ("Third normal form (3NF)", True),
                ("First normal form (1NF)", False),
                ("Second normal form (2NF)", False),
                ("Fourth normal form (4NF)", False),
            ],
        ),
        (
            "Which SQL statement removes a table definition together with all of "
            "its data from the database?",
            [("DROP TABLE", True), ("DELETE FROM", False), ("TRUNCATE TABLE", False), ("ALTER TABLE", False)],
        ),
        (
            "In database transactions, the ACID properties stand for:",
            [
                ("Atomicity, Consistency, Isolation, Durability", True),
                ("Accuracy, Concurrency, Integrity, Distribution", False),
                ("Atomicity, Concurrency, Indexing, Durability", False),
                ("Availability, Consistency, Isolation, Distribution", False),
            ],
        ),
    ],
}

APTITUDE_EXAM = {
    "title": "Aptitude & Reasoning",
    "description": (
        "Quantitative aptitude and logical reasoning MCQs: number series, "
        "percentages, time-speed-distance, coding and blood relations."
    ),
    "duration_minutes": 15,
    "questions": [
        (
            "If 12 pens cost Rs. 96, what is the cost of 25 pens at the same rate?",
            [("Rs. 200", True), ("Rs. 180", False), ("Rs. 210", False), ("Rs. 225", False)],
        ),
        (
            "Find the next number in the series: 2, 6, 12, 20, 30, ?",
            [("42", True), ("40", False), ("44", False), ("46", False)],
        ),
        (
            "A train 240 metres long crosses a pole in 12 seconds. What is its "
            "speed in km/h?",
            [("72 km/h", True), ("60 km/h", False), ("66 km/h", False), ("80 km/h", False)],
        ),
        (
            "In a certain code, CAT is written as 3120. How is DOG written in the "
            "same code?",
            [("4157", True), ("4715", False), ("4517", False), ("4175", False)],
        ),
        (
            "The average of five consecutive odd numbers is 27. What is the "
            "largest of these numbers?",
            [("31", True), ("29", False), ("33", False), ("35", False)],
        ),
        (
            "What is the angle between the hour and minute hands of a clock at "
            "3:15?",
            [("7.5 degrees", True), ("0 degrees", False), ("15 degrees", False), ("30 degrees", False)],
        ),
        (
            "Pointing to a photograph, Ram says, 'She is the daughter of my "
            "grandfather's only son.' How is the woman related to Ram?",
            [("Sister", True), ("Cousin", False), ("Mother", False), ("Aunt", False)],
        ),
        (
            "What is 15% of 480?",
            [("72", True), ("68", False), ("70", False), ("76", False)],
        ),
    ],
}


class Command(BaseCommand):
    help = (
        "Seed demo data: instructor + student accounts and two active MCQ exams. "
        "Idempotent — reuses existing rows on re-run."
    )

    @transaction.atomic
    def handle(self, *args, **options):
        instructor, instructor_created = self._ensure_user(
            username="instructor",
            password="instructor123",
            first_name="Asha",
            last_name="Verma",
            email="instructor@raim.local",
            is_staff=True,
            role="instructor",
        )
        student, student_created = self._ensure_user(
            username="student1",
            password="student123",
            first_name="Rahul",
            last_name="Deshmukh",
            email="student1@raim.local",
            is_staff=False,
            role="student",
        )

        exam_summaries = []
        for spec in (CS_EXAM, APTITUDE_EXAM):
            exam, created = Exam.objects.get_or_create(
                title=spec["title"],
                defaults={
                    "description": spec["description"],
                    "duration_minutes": spec["duration_minutes"],
                    "is_active": True,
                    "created_by": instructor,
                },
            )
            questions_added = self._seed_questions(exam, spec["questions"])
            exam_summaries.append(
                (exam, created, questions_added, len(spec["questions"]))
            )

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("RAIM demo data seeded"))
        self.stdout.write(
            "  Instructor : username 'instructor'  password 'instructor123'  "
            f"({'created' if instructor_created else 'already existed'}, staff access)"
        )
        self.stdout.write(
            "  Student    : username 'student1'    password 'student123'    "
            f"({'created' if student_created else 'already existed'})"
        )
        self.stdout.write(
            "  Note: student1 has NO face enrolled yet — the student must "
            "register their face via the webcam UI (dashboard) before an exam "
            "can be started."
        )
        for exam, created, added, total in exam_summaries:
            self.stdout.write(
                f"  Exam       : '{exam.title}' — {exam.duration_minutes} min, "
                f"{exam.questions.count()} questions, {exam.total_marks} marks "
                f"({'created' if created else 'already existed'}, "
                f"{added}/{total} questions added this run)"
            )
        self.stdout.write(self.style.SUCCESS("Done."))

    def _ensure_user(self, username, password, first_name, last_name, email,
                     is_staff, role):
        """Get or create a demo user plus their Profile. Returns (user, created)."""
        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "is_staff": is_staff,
            },
        )
        if created:
            user.set_password(password)
            user.save()
        elif is_staff and not user.is_staff:
            user.is_staff = True
            user.save(update_fields=["is_staff"])

        profile, _ = Profile.objects.get_or_create(
            user=user, defaults={"role": role}
        )
        if profile.role != role:
            profile.role = role
            profile.save(update_fields=["role"])
        return user, created

    def _seed_questions(self, exam, question_specs):
        """Create the exam's questions and choices; returns how many were new."""
        added = 0
        for order, (text, choices) in enumerate(question_specs, start=1):
            question, created = Question.objects.get_or_create(
                exam=exam,
                order=order,
                defaults={"text": text, "marks": 1},
            )
            if created:
                added += 1
            for choice_text, is_correct in choices:
                Choice.objects.get_or_create(
                    question=question,
                    text=choice_text,
                    defaults={"is_correct": is_correct},
                )
        return added
