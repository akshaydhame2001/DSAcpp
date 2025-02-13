#include <bits/stdc++.h>
using namespace std;

int fibo(int n)
{
    if (n <= 1)
    {
        return n;
    }
    return fibo(n - 1) + fibo(n - 2);
}

int main()
{
    int n;
    cin >> n;
    cout << fibo(n) << endl;
}

// Time complexity: 2^n

// int fib(int n) {
//     if(n == 0) return 0;
//     if(n == 1) return 1;
//     int first = 0, second = 1;
//     for(int i = 2; i <= n; i++){
//         int temp = second + first;
//         first = second;
//         second = temp;
//     }
//     return second;
// }