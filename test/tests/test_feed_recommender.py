from datetime import datetime, timezone
import unittest

from feed.recommender import FeedCandidate, RepositoryProfile, recommend_feed


class RecommendFeedTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 13, tzinfo=timezone.utc)
        self.repositories = [
            RepositoryProfile(
                full_name="ejasonos/agent-harness",
                description="Python harness connecting cloud and local AI models",
                language="Python",
                topics=("agents", "llm"),
            ),
            RepositoryProfile(
                full_name="ejasonos/vcu-software",
                description="Vehicle control software for an electric vehicle",
                language="TypeScript",
                topics=("ev", "embedded"),
            ),
        ]

    def test_related_and_recent_item_ranks_above_unrelated_item(self):
        items = [
            FeedCandidate(
                url="https://github.com/example/agent/issues/1",
                title="Add local model support to an agent harness",
                body="Python project for cloud and local LLMs",
                language="Python",
                created_at=self.now,
            ),
            FeedCandidate(
                url="https://github.com/example/design/issues/2",
                title="Update the design system",
                language="CSS",
                created_at=self.now,
            ),
        ]
        ranked = recommend_feed(items, self.repositories, now=self.now)
        self.assertIn("agent harness", ranked[0].candidate.title)
        self.assertTrue(any("Python" in reason for reason in ranked[0].reasons))

    def test_github_repository_responses_are_normalized(self):
        ranked = recommend_feed(
            [{
                "html_url": "https://github.com/example/project/issues/3",
                "title": "Improve EV controller",
                "body": "TypeScript vehicle control software",
                "created_at": "2026-11-12T12:00:00Z",
                "repository": {"full_name": "example/project", "language": "TypeScript"},
            }],
            [{
                "full_name": "ejasonos/vcu-software",
                "description": "Vehicle control software in an EV",
                "language": "TypeScript",
                "topics": ["ev", "embedded"],
            }],
            now=self.now,
        )
        self.assertEqual(len(ranked), 1)
        self.assertTrue(ranked[0].generation_prompt.startswith("You are a coding assistant"))
        self.assertIn("ejasonos/vcu-software", ranked[0].generation_prompt)

    def test_empty_and_limit_inputs(self):
        self.assertEqual(recommend_feed([], self.repositories), [])
        self.assertEqual(recommend_feed([FeedCandidate(url="", title="A")], [], limit=0), [])

    def test_unknown_dates_are_safe(self):
        ranked = recommend_feed([FeedCandidate(url="", title="General update")], [], now=self.now)
        self.assertEqual(len(ranked), 1)
        self.assertGreaterEqual(ranked[0].score, 0)


if __name__ == "__main__":
    unittest.main()
