using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class SegmentEnum : Enum<Segment>
    {
        public SegmentEnum(IEnumerable<Segment> segments)
        {
            Comparer = new SegmentComparer();
            AddRange(segments.ToList());
        }

        public SegmentEnum()
        {
            Comparer = new SegmentComparer();
        }
        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}