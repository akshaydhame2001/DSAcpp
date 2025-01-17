/*
Given a positive integer n, count the number of digits in n that divide n evenly (i.e., without leaving a remainder).
Return the total number of such digits.

A digit d of n divides n evenly if the remainder when n is divided by d is 0 (n % d == 0).
Digits of n should be checked individually. If a digit is 0, it should be ignored because division by 0 is undefined.
*/

// Function to count the number of digits in n that evenly divide n

#include <bits/stdc++.h>
using namespace std;

int evenlyDivides(int n)
{
    // code here
    int count = 0;
    int number = n;
    while (number > 0)
    {
        int lastDigit = number % 10;
        number = number / 10;
        if (lastDigit != 0 && n % lastDigit == 0)
        {
            count += 1;
        }
    }
    return count;
}

int main()
{
    int n;
    cin >> n;
    cout << evenlyDivides(n) << endl;
}
