using System.Collections.Generic;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class PathCollection : SortedStackListQueue<Path>
    {
        private static readonly PathComparer PathComparer = new PathComparer();

        public PathCollection(IEnumerable<Path> paths)
            : base(paths)
        {
            Comparer = PathComparer;
        }

        public PathCollection()
        {
            Comparer = PathComparer;
        }

        public PathCollection(Path path)
            : base(path)
        {
            Comparer = PathComparer;
        }
    }
}