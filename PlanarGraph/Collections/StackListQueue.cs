using System.Collections.Generic;
using System.Linq;

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
            return collection.All(Contains);
        }

        public virtual bool BelongsTo(StackListQueue<T> collection)
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

        public override bool Equals(object obj)
        {
            var collection = obj as StackListQueue<T>;
            return collection != null && Count == collection.Count &&
                   this.SequenceEqual(collection);
        }

        public override int GetHashCode()
        {
            return this.Aggregate(0,
                (current, item) => (current << 1) ^ (current >> (8*sizeof (int) - 1)) ^ item.GetHashCode());
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