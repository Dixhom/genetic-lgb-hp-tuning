from __future__ import annotations

from typing import TypeVar, List, Dict
from random import choices, random, randrange, shuffle
from heapq import nlargest
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
import Chromosome

# C = TypeVar('C', bound=Chromosome)

class GeneticAlgorithm:

    SelectionType = int
    SELECTION_TYPE_ROULETTE_WHEEL: SelectionType = 1
    SELECTION_TYPE_TOURNAMENT: SelectionType = 2

    def __init__(
            self, initial_population: List[Chromosome],
            threshold: float,
            max_generations: int, mutation_probability: float,
            crossover_probability: float,
            selection_type: SelectionType) -> None:
        """
        遺伝的アルゴリズムを扱うクラス。

        Parameters
        ----------
        initial_population : list of Chromosome
            最初の世代の個体群（染色体群）。
        threshold : float
            問題解決の判定で利用するしきい値。この値を超える個体が
            発生した時点で計算が終了する。
        max_generations : int
            アルゴリズムで実行する最大世代数。
        mutation_probability : float
            変異確率（0.0～1.0）。
        crossover_probability : float
            交叉確率（0.0～1.0）。
        selection_type : int
            選択方式。以下のいずれかの定数値を指定する。
            - SELECTION_TYPE_ROULETTE_WHEEL
            - SELECTION_TYPE_TOURNAMENT
        """
        self._population: List[Chromosome] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_probability: float = mutation_probability
        self._crossover_probability: float = crossover_probability
        self._selection_type: int = selection_type

    def _exec_roulette_wheel_selection(self) -> List[Chromosome]:
        """
        ルーレット選択を行い、交叉などで利用する2つの個体（染色体）を
        取得する。

        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。選択処理は評価関数
            （fitnessメソッド）による重みが設定された状態でランダムに抽出される。

        Notes
        -----
        評価関数の結果の値が負になる問題には利用できない。
        """

        print('exec roulette wheel selection...')

        weights: List[float] = [
            chromosome.get_fitness() for chromosome in self._population]
        selected_chromosomes: List[Chromosome] = choices(
            self._population, weights=weights, k=2)
        return selected_chromosomes

    def _exec_tournament_selection(self) -> List[Chromosome]:
        """
        トーナメント選択を行い、交叉などで利用するための2つの個体
        （染色体）を取得する。

        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。トーナメント
            用に引数で指定された件数分抽出された中から上位の2つの個体が
            設定される。
        """
        participants_num: int = len(self._population) // 2
        participants: List[Chromosome] = choices(self._population, k=participants_num)
        selected_chromosomes: List[Chromosome] = nlargest(n=2, iterable=participants)
        return selected_chromosomes

    def _to_next_generation(self) -> None:
        """
        次世代の個体（染色体）を生成し、個体群の属性値を生成した
        次世代の個体群で置換する。
        """
        new_population: List[Chromosome] = []

        # 元の個体群の件数が奇数件数の場合を加味して件数の比較は等値ではなく
        # 小なりの条件で判定する。
        while len(new_population) < len(self._population):
            parents: List[Chromosome] = self._get_parents_by_selection_type()
            next_generation_chromosomes: List[Chromosome] = \
                self._get_next_generation_chromosomes(parents=parents)
            new_population.extend(next_generation_chromosomes)

        # 2件ずつ次世代のリストを増やしていく都合、元のリストよりも件数が
        # 多い場合は1件リストから取り除いてリストの件数を元のリストと一致させる。
        if len(new_population) > len(self._population):
            del new_population[0]

        self._population = new_population

    def _get_next_generation_chromosomes(
            self, parents: List[Chromosome]) -> List[Chromosome]:
        """
        算出された親の2つの個体のリストから、次世代として扱う
        2つの個体群のリストを取得する。
        一定確率で交叉や変異させ、確率を満たさない場合には引数の値が
        そのまま次世代として設定される。

        Parameters
        ----------
        parents : list of Chromosome
            算出された親の2つの個体のリスト

        Returns
        -------
        next_generation_chromosomes : list of Chromosome
            次世代として設定される、2つの個体を格納したリスト。
        """
        random_val: float = random()
        next_generation_chromosomes: List[Chromosome] = parents
        if random_val < self._crossover_probability:
            next_generation_chromosomes = parents[0].exec_crossover(
                other=parents[1])

        random_val = random()
        if random_val < self._mutation_probability:
            for chromosome in next_generation_chromosomes:
                chromosome.mutate()
        return next_generation_chromosomes

    def _get_parents_by_selection_type(self) -> List[Chromosome]:
        """
        選択方式に応じた親の2つの個体（染色体）のリストを取得する。

        Returns
        -------
        parents : list of Chromosome
            取得された親の2つの個体（染色体）のリスト。

        Raises
        ------
        ValueError
            対応していない選択方式が指定された場合。
        """
        if self._selection_type == self.SELECTION_TYPE_ROULETTE_WHEEL:
            parents: List[Chromosome] = self._exec_roulette_wheel_selection()
        elif self._selection_type == self.SELECTION_TYPE_TOURNAMENT:
            parents = self._exec_tournament_selection()
        else:
            raise ValueError(
                '対応していない選択方式が指定されています : %s'
                % self._selection_type)
        return parents

    def run_algorithm(self) -> Chromosome:
        """
        遺伝的アルゴリズムを実行し、実行結果の個体（染色体）のインスタンス
        を取得する。

        Returns
        -------
        betst_chromosome : Chromosome
            アルゴリズム実行結果の個体。評価関数でしきい値を超えた個体
            もしくはしきい値を超えない場合は指定された世代数に達した
            時点で一番評価関数の値が高い個体が設定される。
        """
        best_chromosome: Chromosome = \
            deepcopy(self._get_best_chromosome_from_population())
        history = [deepcopy(best_chromosome)]
        print('遺伝的アルゴリズムを実行します')
        for generation_idx in tqdm(range(self._max_generations)):
            print(datetime.now())
            print(f' 世代数 : {generation_idx}')
            print(f' 最良個体情報 : \n{best_chromosome}')

            if best_chromosome.get_fitness() >= self._threshold:
                return best_chromosome

            self._to_next_generation()

            currrent_generation_best_chromosome: Chromosome = \
                self._get_best_chromosome_from_population()
            current_gen_best_fitness: float = \
                currrent_generation_best_chromosome.get_fitness()
            if current_gen_best_fitness > best_chromosome.get_fitness():
                best_chromosome = deepcopy(currrent_generation_best_chromosome)

            history.append(deepcopy(best_chromosome)) # keep track of bests

        return best_chromosome, history

    def _get_best_chromosome_from_population(self) -> Chromosome:
        """
        個体群のリストから、評価関数の値が一番高い個体（染色体）を
        取得する。

        Returns
        -------
        best_chromosome : Chromosome
            リスト内の評価関数の値が一番高い個体。
        """

        print('get best chromosome from population...')

        best_chromosome: Chromosome = self._population[0]
        for chromosome in self._population:
            if best_chromosome.get_fitness() < chromosome.get_fitness():
                best_chromosome = chromosome
        return best_chromosome