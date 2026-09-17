/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The station's state is live by definition. Nothing here is worth caching,
  // and a cached "now playing" is worse than no "now playing".
  poweredByHeader: false,
};

export default nextConfig;
