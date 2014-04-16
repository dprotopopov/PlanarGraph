using System.Collections.Generic;
using System.Linq;
using MyCudafy.Collections;
using MyLibrary.Types;

namespace PlanarGraph.GF2
{
    public class BooleanVector : StackListQueue<bool>
    {
        public BooleanVector(IEnumerable<bool> bools) : base(bools)
        {
        }

        public BooleanVector()
        {
        }

        public override string ToString()
        {
            return string.Join((string) "", (IEnumerable<string>) this.Select(b => b ? "1" : "0"));
        }

        public static BooleanVector And(BooleanVector vector1, BooleanVector vector2)
        {
            int count = vector2.Count;
            return new BooleanVector(vector1.Select((b, index) => b && index < count && vector2[index]));
        }

        public static BooleanVector Xor(BooleanVector vector1, BooleanVector vector2)
        {
            int count = vector2.Count;
            var vector = new BooleanVector(vector1.Select((b, index) => Boolean.Xor(b, index < count && vector2[index])));
            if (vector2.Count > vector1.Count)
                vector.AddRange(vector2.GetRange(vector1.Count, vector2.Count - vector1.Count));
            return vector;
        }

        public static int Module(BooleanVector vector1, BooleanVector vector2)
        {
            int count = vector2.Count;
            return vector1.Where((b, index) => b && index < count && vector2[index]).Count();
        }

        public static int Module(BooleanVector vector1)
        {
            return vector1.Where((b, index) => b).Count();
        }

        public bool IsZero()
        {
            return this.All(b => !b);
        }
    }
}