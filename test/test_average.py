from unittest import TestCase

import Average

class TestAverage(TestCase):

  def test_average(self):
        self.assertEqual(Average.average([20, 30, 70]), 40.0)
        self.assertEqual(round(Average.average([1, 5, 7]), 1), 4.3)
        with self.assertRaises(ZeroDivisionError):
            Average.average([])
        with self.assertRaises(TypeError):
            Average.average(20, 30, 70)