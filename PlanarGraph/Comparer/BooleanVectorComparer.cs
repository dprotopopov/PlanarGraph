using System.Collections.Generic;
using MyMath.GF2;

namespace PlanarGraph.Comparer
{
    public class BooleanVectorComparer : IComparer<BooleanVector>
    {
        public int Compare(BooleanVector x, BooleanVector y)
        {
            return x.IndexOf(true) - y.IndexOf(true);
        }
    }
}