/**
 * The rotating rubber stamp in the corner.
 *
 * Pure decoration, and unashamed about it. A page like this needs one thing
 * on it that no dashboard template would ever produce.
 */
export default function Stamp() {
  const words = "no ads · no algorithm · one listener ·";
  // The ring is r=44, so one trip round it is 2*pi*44. Setting textLength to
  // exactly that makes the phrase fill the circle once and meet itself
  // cleanly -- letter-spacing alone either leaves a gap or runs the last word
  // into the first.
  const circumference = 2 * Math.PI * 44;
  return (
    <div className="stamp" aria-hidden="true">
      <svg viewBox="0 0 120 120">
        <defs>
          <path
            id="stamp-ring"
            d="M60,60 m-44,0 a44,44 0 1,1 88,0 a44,44 0 1,1 -88,0"
          />
        </defs>
        <circle cx="60" cy="60" r="57" fill="none" stroke="#2a2427" strokeWidth="2" />
        <circle cx="60" cy="60" r="34" fill="none" stroke="#2a2427" strokeWidth="2" />
        <text textLength={circumference} lengthAdjust="spacing">
          <textPath href="#stamp-ring">{words}</textPath>
        </text>
      </svg>
      <div className="stamp__core">FM</div>
    </div>
  );
}
