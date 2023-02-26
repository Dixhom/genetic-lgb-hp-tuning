from __future__ import annotations

from typing import TypeVar, List, Dict
from abc import ABC, abstractmethod

class Chromosome(ABC):
    """
    染色体（遺伝的アルゴリズムの要素1つ分）を扱う抽象クラス。
    """

    @abstractmethod
    def get_fitness(self) -> float:
        """
        対象の問題に対する染色体の優秀さを取得する評価関数Y用の
        抽象メソッド。

        Returns
        -------
        fitness : float
            対象の問題に対する染色体の優秀さの値。高いほど問題に
            適した染色体となる。
            遺伝的アルゴリズムの終了判定などにも使用される。
        """
        ...

    @classmethod
    @abstractmethod
    def make_random_instance(cls) -> Chromosome:
        """
        ランダムな特徴（属性値）を持ったインスタンスを生成する
        抽象メソッド。

        Returns
        -------
        instance : Chromosome
            生成されたインスタンス。
        """
        ...

    @abstractmethod
    def mutate(self) -> None:
        """
        染色体を（突然）変異させる処理の抽象メソッド。
        インスタンスの属性などのランダムな別値の設定などが実行される。
        """
        ...

    @abstractmethod
    def exec_crossover(self, other: Chromosome) -> List[Chromosome]:
        """
        引数に指定された別の個体を参照し交叉を実行する。

        Parameters
        ----------
        other : Chromosome
            交叉で利用する別の個体。

        Returns
        -------
        result_chromosomes : list of Chromosome
            交叉実行後に生成された2つの個体（染色体）。
        """
        ...

    def __lt__(self, other: Chromosome) -> bool:
        """
        個体間の比較で利用する、評価関数の値の小なり比較用の関数。

        Parameters
        ----------
        other : Chromosome
            比較対象の他の個体。

        Returns
        -------
        result_bool : bool
            小なり条件を満たすかどうかの真偽値。
        """
        return self.get_fitness() < other.get_fitness()