#include <bits/stdc++.h>
using namespace std;

int main()
{
    int n;
    cout << "Enter array size: ";
    cin >> n;
    int arr[n];
    cout << "Enter array elements: ";
    unordered_map<int, int> mpp;
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
        mpp[arr[i]]++;
    }

    // precompute
    // map<int, int> mpp;
    // for (int i = 0; i < n; i++)
    // {
    //     mpp[arr[i]]++;
    // }

    for (auto it : mpp)
    {
        cout << it.first << "->" << it.second << endl;
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

 1. Brute Force O(n*n), O(1) space
 2. Array Hashing O(n*1), O(max_element) space
 3. Map Hashing O(n*logn), O(U) space, U = unique elements
 NOTE: any data type or data structure as key e.g. int, char, long, pair
 4. unordered_map O(n*1) best and avg, O(n*n) worst
 NOTE: only individual data type as key e.g, int, char, long
collisions: 2 or more element at same index stored as linked_list
-division method: query_element(arr[i]) % hash_size (10^6)
-folding method
-mid sqaure method
NOTE: use unordered_map and when timelimit exceeds use map only.
*/