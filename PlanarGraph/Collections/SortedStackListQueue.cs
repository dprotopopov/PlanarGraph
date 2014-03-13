using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;

namespace PlanarGraph.Collections
{
    public class SortedStackListQueue<T> : StackListQueue<T>
    {
        #region

        public SortedStackListQueue(IEnumerable<T> value) : base(value)
        {
        }

        public SortedStackListQueue(T value)
            : base(value)
        {
        }

        public SortedStackListQueue()
        {
        }

        #endregion

        public IComparer<T> Comparer { get; set; }

        #region

        public override bool IsSorted(IEnumerable<T> collection)
        {
            //Debug.WriteLine("Begin {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            List<T> list = collection.ToList();
            //Debug.WriteLine("End {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            return list.Count < 2 ||
                   Enumerable.Range(0, list.Count - 1)
                       .All(i => Comparer.Compare(list[i], list[i + 1]) <= 0);
        }

        public IEnumerable<T> Intersect(IEnumerable<T> array)
        {
            //Debug.WriteLine("Begin {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            var stackListQueue = new StackListQueue<T>();
            var arrays = new StackListQueue<List<T>>(this) {array.ToList()};
            foreach (var arr in arrays.Where(arr => !IsSorted(arr)))
                arr.Sort(Comparer);
            List<int> indexes = arrays.Select((a, i) => 0).ToList();
            List<int> counts = arrays.Select((a, i) => a.Count).ToList();
            MethodInfo methodInfo = Comparer.GetType().GetMethod("Compare");
            while (Enumerable.Range(0, 2).All(i => indexes[i] < counts[i]))
            {
                var value =
                    (int) methodInfo.Invoke(Comparer, arrays.Select((a, i) => a[indexes[i]]).Cast<object>().ToArray());
                if (value == 0)
                {
                    stackListQueue.Add(arrays[0][indexes[0]++]);
                    indexes[1]++;
                }
                else if (value < 0) indexes[0]++;
                else indexes[1]++;
            }
            //Debug.WriteLine("End {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            return stackListQueue;
        }

        public IEnumerable<T> Except(IEnumerable<T> array)
        {
            //Debug.WriteLine("Begin {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            var stackListQueue = new StackListQueue<T>();
            var arrays = new StackListQueue<List<T>>(this) {array.ToList()};
            foreach (var arr in arrays.Where(arr => !IsSorted(arr)))
                arr.Sort(Comparer);
            List<int> indexes = arrays.Select((a, i) => 0).ToList();
            List<int> counts = arrays.Select((a, i) => a.Count).ToList();
            MethodInfo methodInfo = Comparer.GetType().GetMethod("Compare");
            while (Enumerable.Range(0, 2).All(i => indexes[i] < counts[i]))
            {
                var value =
                    (int) methodInfo.Invoke(Comparer, arrays.Select((a, i) => a[indexes[i]]).Cast<object>().ToArray());
                if (value == 0)
                {
                    indexes[0]++;
                    indexes[1]++;
                }
                else if (value < 0) stackListQueue.Add(arrays[0][indexes[0]++]);
                else indexes[1]++;
            }
            stackListQueue.AddRange(arrays.Where((a, i) => i == 0).SelectMany((a, i) => indexes[i] < counts[i]
                ? a.GetRange(indexes[i], counts[i] - indexes[i])
                : new List<T>()));
            //Debug.WriteLine("End {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            return stackListQueue;
        }

        public IEnumerable<T> Union(IEnumerable<T> array)
        {
            //Debug.WriteLine("Begin {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            var stackListQueue = new StackListQueue<T>();
            var arrays = new StackListQueue<List<T>>(this) {array.ToList()};
            foreach (var arr in arrays.Where(arr => !IsSorted(arr)))
                arr.Sort(Comparer);
            List<int> indexes = arrays.Select((a, i) => 0).ToList();
            List<int> counts = arrays.Select((a, i) => a.Count).ToList();
            MethodInfo methodInfo = Comparer.GetType().GetMethod("Compare");
            while (Enumerable.Range(0, 2).All(i => indexes[i] < counts[i]))
            {
                var value =
                    (int) methodInfo.Invoke(Comparer, arrays.Select((a, i) => a[indexes[i]]).Cast<object>().ToArray());
                if (value == 0)
                {
                    stackListQueue.Add(arrays[0][indexes[0]++]);
                    indexes[1]++;
                }
                else if (value < 0) stackListQueue.Add(arrays[0][indexes[0]++]);
                else stackListQueue.Add(arrays[1][indexes[1]++]);
            }
            stackListQueue.AddRange(arrays.SelectMany((a, i) => indexes[i] < counts[i]
                ? a.GetRange(indexes[i], counts[i] - indexes[i])
                : new List<T>()));
            //Debug.WriteLine("End {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            return stackListQueue;
        }


        public new IEnumerable<T> Distinct()
        {
            //Debug.WriteLine("Begin {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            var stackListQueue = new StackListQueue<T>();
            if (Count == 0) return stackListQueue;
            var arrays = new StackListQueue<List<T>>(this);
            foreach (var arr in arrays.Where(arr => !IsSorted(arr)))
                arr.Sort(Comparer);
            List<int> indexes = arrays.Select((a, i) => 0).ToList();
            List<int> counts = arrays.Select((a, i) => a.Count).ToList();
            MethodInfo methodInfo = Comparer.GetType().GetMethod("Compare");
            while (Enumerable.Range(0, 1).All(i => indexes[i] < counts[i] - 1))
            {
                var value =
                    (int)
                        methodInfo.Invoke(Comparer,
                            arrays.SelectMany((a, i) => new StackListQueue<T>(a[indexes[i]]) {a[indexes[i] + 1]})
                                .Cast<object>()
                                .ToArray());
                if (value == 0)
                    indexes[0]++;
                else stackListQueue.Add(arrays[0][indexes[0]++]);
            }
            stackListQueue.Add(arrays[0][counts[0] - 1]);
            //Debug.WriteLine("End {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            return stackListQueue;
        }

        #endregion

        public override bool Equals(object obj)
        {
            var collection = obj as SortedStackListQueue<T>;
            if (collection == null) return false;
            if (!IsSorted(this)) Sort(Comparer);
            if (!IsSorted(collection)) collection.Sort(Comparer);
            return this.SequenceEqual(collection);
        }

        public new void Remove(T item)
        {
            if (!IsSorted(this)) Sort(Comparer);
            int index = BinarySearch(item, Comparer);
            RemoveAt(index);
        }

        public new int IndexOf(T item)
        {
            if (!IsSorted(this)) Sort(Comparer);
            int index = BinarySearch(item, Comparer);
            return index;
        }

        public override int GetHashCode()
        {
            if (!IsSorted(this)) Sort(Comparer);
            return base.GetHashCode();
        }

        public override void AddRangeExcept(IEnumerable<T> value)
        {
            var list = new SortedStackListQueue<T>(value) {Comparer = Comparer};
            AddRange(list.Except(this));
        }

        public void RemoveRange(IEnumerable<T> value)
        {
            if (!IsSorted(this)) Sort(Comparer);
            var list = new StackListQueue<int>(value.Select(v => BinarySearch(v, Comparer))
                .Where(index => index >= 0));
            list.Sort();
            list.Reverse();
            foreach (int index in list) RemoveAt(index);
        }

        public new bool Contains(IEnumerable<T> collection)
        {
            if (!IsSorted(this)) Sort(Comparer);
            IEnumerable<int> indexes = collection.Select(item => BinarySearch(item, Comparer));
            return indexes.All(index => index >= 0);
        }

        public new bool Contains(T item)
        {
            if (!IsSorted(this)) Sort(Comparer);
            int index = BinarySearch(item, Comparer);
            return index >= 0;
        }
    }
}