#include <bits/stdc++.h> // Includes all standard libraries
using namespace std;

// Function to demonstrate basic printing
void print()
{
    cout << "[LOG] Executing print function...\n";
    cout << "Akshay" << endl;
}

// Function to calculate the sum of two integers
int sum(int a, int b)
{
    cout << "[LOG] Executing sum function with inputs: " << a << ", " << b << endl;
    return a + b;
}

// Function to explain the usage of pairs
void explainPair()
{
    cout << "[LOG] Executing explainPair function...\n";

    // Basic pair
    pair<int, int> p1 = {1, 3};
    cout << "[INFO] Pair p1: (" << p1.first << ", " << p1.second << ")" << endl;

    // Nested pair (pair containing another pair)
    pair<int, pair<int, int>> p2 = {1, {3, 4}};
    cout << "[INFO] Nested Pair p2: (" << p2.first << ", ("
         << p2.second.first << ", " << p2.second.second << "))" << endl;

    // Array of pairs
    pair<int, int> arr[] = {{1, 2}, {2, 5}, {5, 1}};
    cout << "[INFO] Array of pairs:\n";
    for (int i = 0; i < 3; i++)
    {
        cout << " - Pair " << i + 1 << ": (" << arr[i].first << ", " << arr[i].second << ")\n";
    }
    cout << "[INFO] Accessing specific pair element: arr[1].second = " << arr[1].second << endl;
}

// Main function
int main()
{
    cout << "[LOG] Program started...\n";

    // Call the print function
    print();

    // Call the sum function and log the result
    int result = sum(1, 5);
    cout << "[RESULT] Sum of 1 and 5: " << result << endl;

    // Call the explainPair function
    explainPair();

    // Demonstrate input from the user
    cout << "[INPUT] Enter an integer value: ";
    int a;
    cin >> a;
    std::cout << "[RESULT] You entered: " << a << endl;

    std::cout << "[LOG] Program ended...\n";
    return 0;
}

/*
[LOG] Program started...
[LOG] Executing print function...
Akshay
[LOG] Executing sum function with inputs: 1, 5
[RESULT] Sum of 1 and 5: 6
[LOG] Executing explainPair function...
[INFO] Pair p1: (1, 3)
[INFO] Nested Pair p2: (1, (3, 4))
[INFO] Array of pairs:
 - Pair 1: (1, 2)
 - Pair 2: (2, 5)
 - Pair 3: (5, 1)
[INFO] Accessing specific pair element: arr[1].second = 5
[INPUT] Enter an integer value: 2
[RESULT] You entered: 2
[LOG] Program ended...
*/
