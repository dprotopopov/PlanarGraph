using System.Collections.Generic;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class PathDictionary : Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection>
    {
        public PathDictionary(IEnumerable<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>> list)
        {
            foreach (var pair in list)
            {
                Add(pair.Key, pair.Value);
            }
        }

        public PathDictionary()
        {
        }

        public PathDictionary(KeyValuePair<Vertex, Vertex> pair, PathCollection pathCollection)
        {
            Add(pair, pathCollection);
        }
    }
}