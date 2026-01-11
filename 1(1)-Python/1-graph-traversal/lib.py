from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        self.graph: DefaultDict[int, list[int]] = defaultdict(list)

    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        """
        self.graph[u].append(v)
        self.graph[v].append(u)
        
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법: 재귀 방식
        - 이미 방문한 노드를 제외하고, 엣지로 연결된 노드를 우선적으로 탐색
        """
        visited_nodes = []

        # 방문 순서를 위해 정렬
        for key in self.graph:
            self.graph[key].sort()
        
        # 재귀 함수 정의
        def recurse(node: int):
            visited_nodes.append(node)
            for next_node in self.graph[node]:
                if next_node not in visited_nodes:
                    recurse(next_node)
        
        recurse(start)
        return visited_nodes
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        
        구현 방법: queue를 사용하여 구현
        - 상위 노드부터 내려가며 큐를 한 칸씩 이동하며 탐색 
        """
        visited_nodes = []

        # 방문 순서를 위해 정렬
        for key in self.graph:
            self.graph[key].sort()
        
        # 큐 정의
        queue: deque[int] = deque()
        queue.append(start)
        visited_nodes.append(start)
        
        # 큐가 빌 때까지 반복
        while queue:
            node = queue.popleft()
            for next_node in self.graph[node]:
                if next_node not in visited_nodes:
                    queue.append(next_node)
                    visited_nodes.append(next_node)
        
        return visited_nodes
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
