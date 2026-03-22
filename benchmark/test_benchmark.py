import pytest
from benchmark import run_benchmark

class TestBenchmark:
    def test_run_benchmark(self):
        result = run_benchmark(100)
        assert "duration" in result
        assert "iterations" in result

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
