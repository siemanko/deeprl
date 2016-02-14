import unittest

from deeprl.settings import update_settings

class SettingsTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(update_settings('siema', None), 'siema')
        self.assertEqual(update_settings(None, 'siema'), 'siema')

    def test_updates(self):
        self.assertEqual(update_settings({'a': 1}, {'b':2}), {'a':1, 'b':2})
        self.assertEqual(update_settings({'a': 1}, {'a':2}), {'a':2})

    def test_immutable(self):
        a = {'a': 1}
        b = {'b': 2}
        self.assertEqual(update_settings(a, b), {'a':1, 'b':2})
        self.assertEqual(a, {'a':1})
        self.assertEqual(b, {'b':2})


if __name__ == '__main__':
    unittest.main()
