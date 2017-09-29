import unittest

class TestFirst(unittest.TestCase):

    def test_adummy(self):
        self.assertEqual(1, 1)
        print "if you've got here, you're done."


# unittest.main() # Calling from the command line invokes all tests
