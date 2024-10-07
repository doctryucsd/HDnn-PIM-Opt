from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Point:
    accuracy: float
    energy: float
    performance: float
    area: float
    idx: int

    def to_list(self) -> List[float]:
        return [self.accuracy, self.energy, self.performance, self.area]

    def to_list_compare(self) -> List[float]:
        # minimize metrics have to be negative for comparsion in is_non_dominated
        return [self.accuracy, -self.energy, -self.performance, -self.area]

    def is_eligible(self, constraints: List[float]) -> bool:
        return (
            self.accuracy >= constraints[0]
            and self.energy <= constraints[1]
            and self.performance <= constraints[2]
            and self.area <= constraints[3]
        )

    def process_plot(self, metrics: List[str]) -> List[float]:
        return [getattr(self, metric) for metric in metrics]

    @staticmethod
    def from_list(lst: List[float], idx: int) -> Point:
        return Point(lst[0], lst[1], lst[2], lst[3], idx)


@dataclass
class PointSet:
    points: List[Point]

    def to_list(self) -> List[List[float]]:
        return [point.to_list() for point in self.points]

    def to_list_compare(self) -> List[List[float]]:
        return [point.to_list_compare() for point in self.points]

    def subset(self, indices: List[int]) -> PointSet:
        return PointSet([self.points[i] for i in indices])

    def exclude(self, indices: List[int]) -> PointSet:
        index_set = set(indices)
        return PointSet(
            [point for i, point in enumerate(self.points) if i not in index_set]
        )

    def union(self, other: PointSet) -> PointSet:
        points: List[Point] = self.points + other.points
        sorted_points = sorted(points, key=lambda x: x.idx)
        return PointSet(sorted_points)

    def get_metric(self, metric: str) -> List[float]:
        return [getattr(point, metric) for point in self.points]

    def get_indices(self) -> List[int]:
        return [point.idx for point in self.points]

    def plot_process(self, metrics: List[str]) -> List[List[float]]:
        """
        Process point set into input to plot_3d_scatter.
        """
        ret: List[List[float]] = []
        for metric in metrics:
            ret.append(self.get_metric(metric))

        assert len(ret) == 3, f"Expected 3 metrics, got {len(ret)}"
        return ret

    @property
    def accuray(self) -> List[float]:
        return [point.accuracy for point in self.points]

    @property
    def energy(self) -> List[float]:
        return [point.energy for point in self.points]

    @property
    def performance(self) -> List[float]:
        return [point.performance for point in self.points]

    @property
    def area(self) -> List[float]:
        return [point.area for point in self.points]

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx):
        if isinstance(idx, int):  # Handle single index
            return self.points[idx]
        elif isinstance(idx, slice):  # Handle slice object
            return PointSet(self.points[idx])
        else:
            raise TypeError(f"Invalid argument type: {type(idx)}")


    def __iter__(self):
        return iter(self.points)

    @staticmethod
    def from_dict(metrics: Dict[str, List[float]]) -> PointSet:
        assert (
            len(metrics["accuracy"])
            == len(metrics["energy"])
            == len(metrics["timing"])
            == len(metrics["area"])
        ), f"Metrics should have the same length, got {len(metrics['accuracy'])}, {len(metrics['energy'])}, {len(metrics['timing'])}, {len(metrics['area'])}"

        points: List[Point] = []
        for i, (accuracy, energy, performance, area) in enumerate(
            zip(
                metrics["accuracy"],
                metrics["energy"],
                metrics["timing"],
                metrics["area"],
            )
        ):
            points.append(Point(accuracy, energy, performance, area, i))

        return PointSet(points)

