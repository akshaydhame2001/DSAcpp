#include <bits/stdc++.h>
using namespace std;

bool isPrime(int num)
{
    if (num <= 1)
        return false; // Numbers <= 1 are not prime
    if (num <= 3)
        return true; // 2 and 3 are prime numbers
    if (num % 2 == 0 || num % 3 == 0)
        return false; // Eliminate multiples of 2 and 3

    // Check for factors from 5 to sqrt(num), skipping even numbers
    for (int i = 5; i * i <= num; i += 6)
    {
        if (num % i == 0 || num % (i + 2) == 0)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    int n;
    cout << "Enter a number: ";
    cin >> n;

    if (isPrime(n))
    {
        cout << n << " is a prime number." << endl;
    }
    else
    {
        cout << n << " is not a prime number." << endl;
    }

    return 0;
}
