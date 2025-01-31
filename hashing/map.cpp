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
    map<int, int> mpp;
    for (int i = 0; i < n; i++)
    {
        mpp[arr[i]]++;
    }

    int q;
    cout << "Number of queries: ";
    cin >> q;
    cout << "Queries: ";
    while (q--)
    {
        int number;
        cin >> number;
        // fetch
        cout << mpp[number] << endl;
    }
    return 0;
}

/*
 Comparison of Approaches for Querying Frequencies

 1. Brute Force (O(Q * N), O(1) space)
    - Check each query by iterating through the array.
    - Pros: Simple, easy to implement.
    - Cons: Very slow for large N, Q.

 2. Array Hashing (O(N + Q), O(max_element) space)
    - Precompute frequencies in an array.
    - Pros: Fast O(1) lookup for queries.
    - Cons: High memory usage if numbers are large (e.g., 10^9).

 3. Map Hashing (O(N + Q), O(U) space, U = unique elements)
    - Store frequencies in a dictionary (hashmap).
    - Pros: Efficient for large/sparse numbers.
    - Cons: Slightly higher constant overhead. (collisions)
*/