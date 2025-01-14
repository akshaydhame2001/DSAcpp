#include <bits/stdc++.h>
using namespace std;

void explainSet()
{
    set<int> st; // Set is a sorted and unique container (no duplicates).

    // Insert elements (O(log n) time complexity).
    st.insert(1);  // {1}
    st.emplace(2); // {1, 2}
    st.insert(2);  // {1, 2} (no duplicate allowed)
    st.insert(4);  // {1, 2, 4}
    st.insert(3);  // {1, 2, 3, 4}
    cout << "After inserting elements: ";
    for (auto x : st)
        cout << x << " ";
    cout << endl;

    // Find an element
    auto it = st.find(3); // Returns iterator pointing to 3.
    if (it != st.end())
    {
        cout << "Element 3 found in the set." << endl;
    }
    else
    {
        cout << "Element 3 not found in the set." << endl;
    }

    auto it_not_found = st.find(6); // If not found, returns st.end(). (address after end ele)
    if (it_not_found == st.end())
    {
        cout << "Element 6 not found in the set." << endl;
    }

    // Erase an element
    st.erase(4); // Removes 4 (O(log n)).
    cout << "After erasing 4: ";
    for (auto x : st)
        cout << x << " ";
    cout << endl;

    // Check if an element exists
    int cnt = st.count(1); // Checks if 1 exists.
    cout << "Count of 1 in the set: " << cnt << endl;

    // Erase using iterator:
    it = st.find(3);
    if (it != st.end())
    {
        st.erase(it); // Removes the element 3. O(1)
        cout << "After erasing 3 using iterator: ";
        for (auto x : st)
            cout << x << " ";
        cout << endl;
    }

    // Erase elements in range [first, last):
    auto it1 = st.find(2);
    auto it2 = st.find(4); // Equivalent to the end of the set.
    st.erase(it1, it2);    // Removes elements from 2 (inclusive) to the end.
    cout << "After erasing elements from 2 to 4 [2,4): ";
    for (auto x : st)
        cout << x << " ";
    cout << endl;

    // Bounds:
    st.insert({1, 2, 3, 4}); // Re-insert elements to test bounds.
    cout << "After re-inserting elements: ";
    for (auto x : st)
        cout << x << " ";
    cout << endl;

    auto lb = st.lower_bound(2); // Points to 2 or next larger element.
    cout << "Lower bound of 2: " << (lb != st.end() ? to_string(*lb) : "Not found") << endl;

    auto ub = st.upper_bound(2); // Points to element strictly greater than 2.
    cout << "Upper bound of 2: " << (ub != st.end() ? to_string(*ub) : "Not found") << endl;

    // Final display of the set:
    cout << "Final set elements: ";
    for (auto x : st)
        cout << x << " ";
    cout << endl;
}

void explainMultiSet()
{
    // Multiset is a sorted container that allows duplicate elements.
    multiset<int> ms;

    // Inserting elements
    ms.insert(1); // {1}
    ms.insert(1); // {1, 1}
    ms.insert(2); // {1, 1, 2}
    cout << "Initial multiset elements: ";
    for (auto x : ms)
        cout << x << " ";
    cout << endl;

    // Erase all occurrences of an element
    ms.erase(1); // Removes all occurrences of 1
    cout << "After erasing all 1's: ";
    for (auto x : ms)
        cout << x << " ";
    cout << endl;

    // Reinsert elements to demonstrate single-element erasure
    ms.insert(1); // {1, 2}
    ms.insert(1); // {1, 1, 2}
    cout << "After reinserting 1's: ";
    for (auto x : ms)
        cout << x << " ";
    cout << endl;

    // Count occurrences of an element
    int cnt = ms.count(1);
    cout << "Count of 1: " << cnt << endl;

    // Erase a single occurrence of an element using an iterator
    auto it = ms.find(1); // Finds the first occurrence of 1
    if (it != ms.end())
    {
        ms.erase(it); // Removes only one occurrence of 1
    }
    cout << "After erasing one occurrence of 1: ";
    for (auto x : ms)
        cout << x << " ";
    cout << endl;

    // Erase a range of elements
    ms.insert(1); // {1, 1, 2}
    ms.insert(3); // {1, 1, 2, 3}
    cout << "After reinserting elements: ";
    for (auto x : ms)
        cout << x << " ";
    cout << endl;
    auto rangeStart = ms.find(1);
    auto rangeEnd = ms.find(2); // Stops just before 2
    if (rangeStart != ms.end() && rangeEnd != ms.end())
    {
        ms.erase(rangeStart, rangeEnd); // Erases all 1's
    }
    cout << "After erasing range [1, 2): ";
    for (auto x : ms)
        cout << x << " ";
    cout << endl;

    // Display final multiset elements
    cout << "Final multiset elements: ";
    for (auto x : ms)
        cout << x << " ";
    cout << endl;
}

void explainUSet()
{
    // unique and unordered
    // O(1) worst case: O(n)
    // LB, UB doesn't work
    unordered_set<int> st;
    st.insert(1);
    st.insert(1);
    st.insert(2);
    cout << "Unordered Set: ";
    for (auto x : st)
    {
        cout << x << " ";
    }
    cout << endl;
}

int main()
{
    explainSet();
    explainMultiSet();
    explainUSet();
    return 0;
}

/*
After inserting elements: 1 2 3 4
Element 3 found in the set.
Element 6 not found in the set.
After erasing 4: 1 2 3
Count of 1 in the set: 1
After erasing 3 using iterator: 1 2
After erasing elements from 2 to 4 [2,4): 1
After re-inserting elements: 1 2 3 4
Lower bound of 2: 2
Upper bound of 2: 3
Final set elements: 1 2 3 4

Initial multiset elements: 1 1 2
After erasing all 1's: 2
After reinserting 1's: 1 1 2
Count of 1: 2
After erasing one occurrence of 1: 1 2
After reinserting elements: 1 1 2 3
After erasing range [1, 2): 2 3
Final multiset elements: 2 3
Unordered Set: 2 1
*/