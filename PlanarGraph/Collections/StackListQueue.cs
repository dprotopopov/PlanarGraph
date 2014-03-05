using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Types;

namespace PlanarGraph.Collections
{
    public class StackListQueue<T> : List<T>
    {
        public IComparer<T> Comparer { get; set; }

        public void Enqueue(T value)
        {
            Add(value);
        }

        public T Dequeue()
        {
            T value = this[0];
            RemoveAt(0);
            return value;
        }

        public void Push(T value)
        {
            Add(value);
        }

        public void Add(IEnumerable<T> value)
        {
            AddRange(value);
        }

        public T Pop()
        {
            int index = Count;
            T value = this[--index];
            RemoveAt(index);
            return value;
        }

        public static StackListQueue<T> IntersectSorted(StackListQueue<T> array1, StackListQueue<T> array2,
            IComparer<T> comparer)
        {
            int i = 0;
            int j = 0;
            var stackListQueue = new StackListQueue<T>();
            while (i < array1.Count && j < array2.Count)
            {
                int value = comparer.Compare(array1[i], array2[j]);
                if (value == 0)
                {
                    stackListQueue.Add(array1[i]);
                    i++;
                    j++;
                }
                else if (value < 0) i++;
                else j++;
            }
            return stackListQueue;
        }

        public static StackListQueue<T> DistinctSorted(StackListQueue<T> array1, StackListQueue<T> array2,
            IComparer<T> comparer)
        {
            int i = 0;
            int j = 0;
            var stackListQueue = new StackListQueue<T>();
            while (i < array1.Count && j < array2.Count)
            {
                int value = comparer.Compare(array1[i], array2[j]);
                if (value != 0)
                {
                    stackListQueue.Add(value < 0 ? array1[i++] : array2[j++]);
                }
                else
                {
                    stackListQueue.Add(array1[i++]);
                    j++;
                }
            }
            if (i < array1.Count)
                stackListQueue.AddRange(array1.GetRange(i, array1.Count - i));
            if (j < array2.Count)
                stackListQueue.AddRange(array2.GetRange(j, array2.Count - j));
            return stackListQueue;
        }

        public bool Contains(IEnumerable<T> collection)
        {
            if (!collection.Select(Contains).Aggregate(true, Boolean.And)) return false;
            int index = IndexOf(collection.First());
            int count = Count;
            return collection.Select((item, index1) => item.Equals(this[(index + index1)%count]))
                .Aggregate(true, Boolean.And) ||
                   collection.Select((item, index1) => item.Equals(this[(index + count - index1)%count]))
                       .Aggregate(true, Boolean.And);
        }

        public override bool Equals(object obj)
        {
            var collection = obj as StackListQueue<T>;
            return collection != null && Count == collection.Count && Contains(collection);
        }

        public override int GetHashCode()
        {
            List<T> list = this.ToList();
            list.Sort(Comparer);
            int index = IndexOf(list.First());
            List<T> vertices = GetRange(index, Count - index);
            vertices.AddRange(GetRange(0, index));
            if (vertices.Count > 1 && Comparer.Compare(vertices[1], vertices[vertices.Count - 1]) > 0)
            {
                vertices.Reverse();
                vertices.Insert(0, vertices.Last());
                vertices.RemoveAt(vertices.Count - 1);
            }
            return vertices.Aggregate(0,
                (current, item) => (current << 1) ^ (current >> (8*sizeof (int) - 1)) ^ item.GetHashCode());
        }

        public override string ToString()
        {
            return string.Join(",", this.Select(item => item.ToString()));
        }
    }
}