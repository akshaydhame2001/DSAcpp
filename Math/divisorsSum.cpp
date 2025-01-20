/*
Given a positive integer n, The task is to find the value of Σi F(i) where i is from 1 to n and
function F(i) is defined as the sum of all divisors of i.
*/
#include <bits/stdc++.h>
using namespace std;

int sumOfDivisors1(int n)
{
    // Write Your Code here
    int sum = 0;
    for (int i = 1; i <= n; i++)
    {
        vector<int> ls;
        for (int j = 1; j <= sqrt(i); j++)
        {
            if (i % j == 0)
            {
                ls.push_back(j);
                if (i / j != j)
                {
                    ls.push_back(i / j);
                }
            }
        }
        for (int k : ls)
        {
            sum = sum + k;
        }
    }
    return sum;
}

// summation of N*(n/N)

int sumOfDivisors2(int n)
{
    int sum = 0;
    for (int i = 1; i <= n; i++)
    {
        int count = n / i;
        sum += i * count;
    }
    return sum;
}

int main()
{
    int n;
    cin >> n;
    cout << sumOfDivisors1(n) << endl;
    cout << sumOfDivisors2(n) << endl;
    return 0;
}