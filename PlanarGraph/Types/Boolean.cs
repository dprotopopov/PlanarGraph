namespace PlanarGraph.Types
{
    public static class Boolean
    {
        public static bool And(bool b, bool b1)
        {
            return b && b1;
        }

        public static bool Or(bool b, bool b1)
        {
            return b || b1;
        }

        public static bool Xor(bool b, bool b1)
        {
            return b ^ b1;
        }
    }
}