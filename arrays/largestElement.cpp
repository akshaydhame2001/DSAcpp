#include <bits/stdc++.h>
using namespace std;

int largest(vector<int> &arr)
{
    // code here
    int largestE = arr[0];
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] > largestE)
        {
            largestE = arr[i];
        }
    }
    return largestE;
}

int main()
{
    int n;
    cin >> n;
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
        cin >> arr[i];
    cout << largest(arr) << endl;
    return 0;
}