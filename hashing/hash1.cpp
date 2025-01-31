#include <bits/stdc++.h>
using namespace std;

int main()
{
    int n;
    cout << "Enter array size: ";
    cin >> n;
    int arr[n];
    cout << "Enter array elements: ";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    // precompute
    int hash[13] = {0};
    for (int i = 0; i < n; i++)
    {
        hash[arr[i]] += 1;
    }

    int q;
    cout << "Number of queries: ";
    cin >> q;
    cout << "Queries: ";
    while (q--)
    {
        int number;
        cin >> number;
        cout << hash[number] << endl;
    }
    return 0;
}

// max array size 10^6 inside int main() for bool 10^7
// max array size 10^7 globally for bool 10^8