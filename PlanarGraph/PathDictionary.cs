using System.Collections.Generic;
using PlanarGraph.Collections;
using PlanarGraph.Data;

namespace PlanarGraph
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
    }
}