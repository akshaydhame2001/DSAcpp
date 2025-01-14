#include <bits/stdc++.h>
using namespace std;

void explainMap()
{
    cout << "Explaining map:" << endl;

    // Initialize different map types
    map<int, int> mpp;                   // A simple map with integer keys and values.
    map<int, pair<int, int>> mppPair;    // A map with integer keys and pair<int, int> as values.
    map<pair<int, int>, int> mppPairMap; // A map with pair<int, int> as keys and integer values.

    // Inserting values into the map
    mpp[1] = 2;
    mpp.emplace(3, 1);
    mpp.insert({2, 4});
    mppPairMap[{2, 3}] = 10;

    // Output the map contents
    cout << "Contents of mpp (key-value pairs):" << endl;
    for (auto it : mpp)
    {
        cout << "Key: " << it.first << ", Value: " << it.second << endl;
    }

    // Accessing elements by key
    cout << "\nAccessing mpp[1]: " << mpp[1] << endl;                                      // Prints 2
    cout << "Accessing mpp[5] (non-existing, default value inserted): " << mpp[5] << endl; // Prints 0

    // Finding an element using 'find'
    auto it = mpp.find(3);
    if (it != mpp.end())
    {
        cout << "\nFound element with key 3: " << it->first << " -> " << it->second << endl;
    }
    else
    {
        cout << "\nElement with key 3 not found." << endl;
    }

    // Lower and Upper Bound
    auto lower = mpp.lower_bound(2); // First element not less than 2
    auto upper = mpp.upper_bound(3); // First element greater than 3

    cout << "\nLower bound for key 2: " << lower->first << " -> " << lower->second << endl;
    cout << "Upper bound for key 3: " << upper->first << " -> " << upper->second << endl;

    cout << endl;
}

void explainMultimap()
{
    // dublicate keys
}

void explainUnorderedMap()
{
    // unique but unordered
    // O(1) worst case: O(n)
}

int main()
{
    explainMap();
    return 0;
}

/*
Contents of mpp (key-value pairs):
Key: 1, Value: 2
Key: 2, Value: 4
Key: 3, Value: 1

Accessing mpp[1]: 2
Accessing mpp[5] (non-existing, default value inserted): 0

Found element with key 3: 3 -> 1

Lower bound for key 2: 2 -> 4
Upper bound for key 3: 5 -> 0
*/