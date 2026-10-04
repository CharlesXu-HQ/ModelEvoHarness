import json
import unittest
from importlib.resources import files


DATA = files("model_evo_harness").joinpath("data")


class MethodCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards_data = json.loads(DATA.joinpath("method_cards.json").read_text())
        cls.inventory = json.loads(DATA.joinpath("upstream_inventory.json").read_text())
        cls.catalog = json.loads(DATA.joinpath("catalog.json").read_text())
        cls.cards = cls.cards_data["method_cards"]

    def test_exactly_one_card_per_pinned_model_module(self):
        paths = [card["source_model"] for card in self.cards]
        self.assertEqual(self.cards_data["upstream_commit"], self.inventory["commit_sha"])
        self.assertEqual(len(paths), 38)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), set(self.inventory["models"]))

    def test_unique_ids_and_valid_family_membership(self):
        ids = [card["id"] for card in self.cards]
        self.assertEqual(len(ids), len(set(ids)))
        families = {family["id"]: set(family["source_models"])
                    for family in self.catalog["families"]}
        for card in self.cards:
            with self.subTest(model=card["id"]):
                self.assertIn(card["family_id"], families)
                self.assertIn(card["source_model"], families[card["family_id"]])

    def test_cards_include_testable_mechanism_and_boundary(self):
        required_text = ("mechanism", "comparison", "failure_signals",
                         "implementation_boundary")
        for card in self.cards:
            with self.subTest(model=card["id"]):
                self.assertEqual(card["id"], card["source_model"].split("/")[-1][:-3])
                self.assertTrue(card["requires"])
                self.assertTrue(all(isinstance(value, str) and value for value in card["requires"]))
                for key in required_text:
                    self.assertTrue(isinstance(card[key], str) and card[key].strip(), key)
                self.assertIn("same", card["comparison"].lower())
                self.assertIn("split", card["comparison"].lower())


if __name__ == "__main__":
    unittest.main()
