import Station from "@/components/Station";
import { readStation } from "@/lib/station";

// The whole page is "what is happening right now", so there is nothing here
// worth rendering ahead of time.
export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function Page() {
  const result = await readStation();

  return (
    <Station
      initial={result.ok ? result.state : null}
      initialError={result.ok ? "" : result.error}
      streamUrl={process.env.STREAM_URL ?? ""}
      embedUrl={process.env.PLAYER_EMBED_URL ?? ""}
    />
  );
}
