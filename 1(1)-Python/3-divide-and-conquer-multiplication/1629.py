# lib.py의 Matrix 클래스를 참조하지 않음
import sys


"""
TODO:
- fast_power 구현하기 
"""


def fast_power(base: int, exp: int, mod: int) -> int:
    """
    분할 정복을 이용한 빠른 거듭제곱 알고리즘 (O(log exp))
    
    Parameters:
        base (int): 밑
        exp (int): 지수
        mod (int): 나눌 수
        
    Returns:
        int: (base ** exp) % mod
        
    Algorithm:
        - 지수가 0이면 1 반환 (base case)
        - 지수가 짝수이면: (half_power) ** 2 % mod
        - 지수가 홀수이면: (half_power) ** 2 * base % mod
    """
    if exp == 0:
        return 1
    # 거듭제곱이 짝수일 때, base를 제곱하고 exp를 2로 나눔 
    elif exp % 2 == 0:
        return fast_power(base, exp // 2, mod) ** 2 % mod
    # 거듭제곱이 홀수일 때, base를 제곱하고 exp를 2로 나눔 후 base를 곱함
    else:
        return fast_power(base, exp // 2, mod) ** 2 % mod * base % mod  
    

def main() -> None:
    A: int
    B: int
    C: int
    A, B, C = map(int, input().split()) # 입력 고정
    
    result: int = fast_power(A, B, C) # 출력 형식
    print(result) 

if __name__ == "__main__":
    main()
