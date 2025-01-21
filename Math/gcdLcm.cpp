#include <bits/stdc++.h>
using namespace std;

vector<int> lcmAndGcd(int a, int b)
{
    // code here
    // 1. dividing both nums till min(a, b)
    // 2. Euclidean algorithm: gcd(a%b, b) till a || b becomes zero where greater%smaller. O(log phi (min(a, b)))
    int gcd = 1;
    int n1 = a;
    int n2 = b;
    while (a > 0 && b > 0)
    {
        if (a > b)
        {
            a = a % b;
        }
        else
        {
            b = b % a;
        }
    }
    if (a == 0)
    {
        gcd = b;
    }
    else if (b == 0)
    {
        gcd = a;
    }
    int lcm = (n1 * n2) / gcd;
    return {lcm, gcd};
}

int main()
{
    int a, b;
    cout << "Enter two numbers: ";
    cin >> a >> b;
    cout << "[LCM, GCD]: ";
    cout << "[" << lcmAndGcd(a, b)[0] << "," << lcmAndGcd(a, b)[1] << "]" << endl;

    return 0;
}