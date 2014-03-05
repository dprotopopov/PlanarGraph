using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Types;

namespace PlanarGraph.Collections
{
    public class Enum<T> : StackListQueue<T>
    {
        public override bool Equals(object obj)
        {
            var collection = obj as Enum<T>;
            if (collection == null) return false;
            Sort(Comparer);
            collection.Sort(Comparer);
            return base.Equals(collection);
        }

        public new void Add(T item)
        {
            if(!Contains(item)) base.Add(item);
        }

        public override int GetHashCode()
        {
            Sort(Comparer);
            return base.GetHashCode();
        }

        public new bool Contains(IEnumerable<T> collection)
        {
            return collection.Select(Contains).Aggregate(true, Boolean.And);
        }
    }
}