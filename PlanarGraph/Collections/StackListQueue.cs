using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using PlanarGraph.Parallel;

namespace PlanarGraph.Collections
{
    public class StackListQueue<T> : List<T>
    {
        #region

        public StackListQueue(IEnumerable<T> value)
        {
            AddRange(value);
        }

        public StackListQueue(T value)
        {
            Add(value);
        }

        public StackListQueue()
        {
        }

        #endregion

        public virtual void Enqueue(T value)
        {
            Add(value);
        }

        public virtual void Enqueue(IEnumerable<T> value)
        {
            AddRange(value);
        }

        public virtual void ReplaceAll(IEnumerable<T> value)
        {
            List<T> list = value.ToList();
            Clear();
            AddRange(list);
        }

        public virtual T Dequeue()
        {
            T value = this[0];
            RemoveAt(0);
            return value;
        }

        public void Rotate()
        {
            base.Add(this[0]);
            base.RemoveAt(0);
        }

        public void Rotate(int count)
        {
            base.AddRange(GetRange(0, count));
            base.RemoveRange(0, count);
        }

        public virtual IEnumerable<T> Dequeue(int count)
        {
            IEnumerable<T> value = GetRange(0, count);
            RemoveRange(0, count);
            return value;
        }

        public virtual IEnumerable<T> GetReverse()
        {
            int count = Count - 1;
            return this.Select((t, i) => this[count - i]);
        }

        public virtual void Push(T value)
        {
            Add(value);
        }

        public virtual void Prepend(T value)
        {
            Insert(0, value);
        }

        public virtual void Add(IEnumerable<T> value)
        {
            AddRange(value);
        }

        public virtual void AddExcept(T item)
        {
            if (!Contains(item)) Add(item);
        }

        public virtual void AddRangeExcept(IEnumerable<T> value)
        {
            AddRange(value.Except(this));
        }

        public virtual T Pop()
        {
            int index = Count;
            T value = this[--index];
            RemoveAt(index);
            return value;
        }

        public virtual IEnumerable<T> Pop(int count)
        {
            int index = Count - count;
            IEnumerable<T> value = GetRange(index, count);
            RemoveRange(index, count);
            return value;
        }

        public virtual void Push(IEnumerable<T> value)
        {
            AddRange(value);
        }

        public bool Contains(IEnumerable<T> collection)
        {
            try
            {
                //Debug.WriteLine("Begin {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
                IEnumerable<IEnumerable<int>> list1 = collection.Select(GetInts);
                IEnumerable<IEnumerable<int>> list2 = this.Select(GetInts);
                int[][] matrix;
                int[] counts;
                lock (CudafySequencies.Semaphore)
                {
                    CudafySequencies.SetSequencies(
                        list1.Select(item => item.ToArray()).ToArray(),
                        list2.Select(item => item.ToArray()).ToArray()
                        );
                    CudafySequencies.Execute("Compare");
                    matrix = CudafySequencies.GetMatrix();
                }
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(matrix);
                    CudafyMatrix.Execute("IndexOfZero");
                    counts = CudafyMatrix.GetCounts();
                }
                //Debug.WriteLine("End {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
                return counts.SequenceEqual(Enumerable.Repeat(1, Count));
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                return collection.All(Contains);
            }
        }

        public IEnumerable<T> Distinct()
        {
            if (Count == 0) return new List<T>();
            IEnumerable<IEnumerable<int>> list = this.Select(GetInts);
            int[][] matrix;
            int[] indexes;
            lock (CudafySequencies.Semaphore)
            {
                int[][] arr = this.Select(GetInts).Select(item => item.ToArray()).ToArray();
                CudafySequencies.SetSequencies(arr, arr);
                CudafySequencies.Execute("Compare");
                matrix = CudafySequencies.GetMatrix();
            }
            lock (CudafyMatrix.Semaphore)
            {
                CudafyMatrix.SetMatrix(matrix);
                CudafyMatrix.Execute("IndexOfZero");
                indexes = CudafyMatrix.GetIndexes();
            }
            return indexes.Where((value, index) => value == index)
                .Select(index => this[index]);
        }

        public virtual bool BelongsTo(IEnumerable<T> collection)
        {
            if (!this.All(collection.Contains)) return false;
            var forward = new StackListQueue<T>(collection);
            for (int index = forward.IndexOf(this[0]);
                index >= 0 && index + Count <= forward.Count();
                index = forward.IndexOf(this[0]))
            {
                if (this.SequenceEqual(forward.GetRange(index, Count))) return true;
                forward.RemoveRange(0, index);
            }
            return false;
        }

        public virtual IEnumerable<int> GetInts(T values)
        {
            throw new NotImplementedException();
        }

        public override bool Equals(object obj)
        {
            var collection = obj as SortedStackListQueue<T>;
            if (collection == null) return false;
            return this.SequenceEqual(collection);
        }

        public virtual bool IsSorted(IEnumerable<T> collection)
        {
            //Debug.WriteLine("Begin {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            if (collection.Count() < 2) return true;
            IEnumerable<IEnumerable<int>> list1 = this.Select(GetInts);
            int[][] matrix;
            int[] indexes;
            lock (CudafySequencies.Semaphore)
            {
                int[][] arr = list1.Select(item => item.ToArray()).ToArray();
                CudafySequencies.SetSequencies(arr, arr);
                CudafySequencies.Execute("Compare");
                matrix = CudafySequencies.GetMatrix();
            }
            lock (CudafyMatrix.Semaphore)
            {
                CudafyMatrix.SetMatrix(matrix);
                CudafyMatrix.Execute("IndexOfNonPositive");
                indexes = CudafyMatrix.GetIndexes();
            }
            //Debug.WriteLine("End {0}::{1}", GetType().Name, MethodBase.GetCurrentMethod().Name);
            return Enumerable.Range(0, indexes.Length - 1).All(i => indexes[i] <= indexes[i + 1]);
        }

        public override int GetHashCode()
        {
            try
            {
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(this.Select(GetInts).Select(item => item.ToArray()).ToArray());
                    CudafyMatrix.ExecuteHash();
                    return CudafyMatrix.GetHash();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                return this.Aggregate(0,
                    (current, item) => (current << 1) ^ (current >> (8*sizeof (int) - 1)) ^ item.GetHashCode());
            }
        }

        public override string ToString()
        {
            return string.Join(",", this.Select(item => item.ToString()));
        }

        public void Append(T value)
        {
            Add(value);
        }
    }
}