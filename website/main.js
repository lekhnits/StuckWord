// Stuck on the Word project page.

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Words in an invented language (nobody speaks it) and their English translations.
const FLASHCARDS = [
  ["velmora", "lantern"],
  ["trisk", "window"],
  ["oudane", "river"],
  ["zelvik", "pocket"],
  ["nuvaro", "thunder"],
];

// Each card is answered the way the planned hint works: blanks while searching, then the
// first letter, then the whole word.
function startFlashcards() {
  const wordElement = document.getElementById("flashcard-word");
  const answerElement = document.getElementById("flashcard-answer");
  if (REDUCED_MOTION) {
    const [word, translation] = FLASHCARDS[0];
    wordElement.textContent = word;
    answerElement.textContent = translation;
    return;
  }

  const steps = FLASHCARDS.flatMap(([word, translation]) => {
    const blanks = "_".repeat(translation.length);
    return [
      { word, answer: "", hold: 900 },
      { word, answer: blanks, hold: 1700 },
      { word, answer: translation[0] + blanks.slice(1), hold: 1300 },
      ...Array.from({ length: translation.length - 1 }, (_, index) => ({
        word,
        answer: translation.slice(0, index + 2) + blanks.slice(index + 2),
        hold: 110,
      })),
      { word, answer: translation, hold: 2400 },
    ];
  });

  let stepIndex = 0;
  function showNextStep() {
    const step = steps[stepIndex];
    wordElement.textContent = step.word;
    answerElement.textContent = step.answer;
    isSearching = step.answer.includes("_");
    stepIndex = (stepIndex + 1) % steps.length;
    setTimeout(showNextStep, step.hold);
  }
  showNextStep();
}

// A sketched brain wave: a slow, larger rhythm while searching, small and calm otherwise.
let isSearching = false;

function startBrainWave() {
  const path = document.getElementById("brain-wave");
  const width = 600;
  const middle = 28;
  let amplitude = 0.2;
  let phase = 0;

  function waveShape(searchLevel) {
    const points = [];
    for (let x = 0; x <= width; x += 4) {
      const slow = Math.sin((x / width) * Math.PI * 2 * 5 - phase);          // theta-like
      const fast = Math.sin((x / width) * Math.PI * 2 * 17 - phase * 2.6);   // background
      const y = middle - (16 * searchLevel * slow + 3.5 * fast);
      points.push(`${x},${y.toFixed(1)}`);
    }
    return `M${points.join(" L")}`;
  }

  if (REDUCED_MOTION) {
    path.setAttribute("d", waveShape(0.6));
    return;
  }
  function frame() {
    amplitude += ((isSearching ? 1 : 0.15) - amplitude) * 0.07;  // ease towards the target
    phase += 0.05;
    path.setAttribute("d", waveShape(amplitude));
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// Vertical neurowaves on both edges of the page; letters of the invented language ride along one
// of them and drift slowly downwards.
function startSideWaves() {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const LETTER_TEXT = `${FLASHCARDS.map(([word]) => word).join(" · ")} · `.repeat(12);
  const WIDTH = 150;
  const sides = [...document.querySelectorAll(".side-wave")].map((svg, sideIndex) => {
    const line = document.createElementNS(SVG_NS, "path");
    line.setAttribute("class", "wave-line");
    line.setAttribute("stroke", "#c9cdeb");
    const carrier = document.createElementNS(SVG_NS, "path");
    carrier.setAttribute("id", `side-wave-path-${sideIndex}`);
    carrier.setAttribute("fill", "none");
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("fill", "#dba889");
    const textPath = document.createElementNS(SVG_NS, "textPath");
    textPath.setAttribute("href", `#side-wave-path-${sideIndex}`);
    textPath.textContent = LETTER_TEXT;
    text.appendChild(textPath);
    svg.append(line, carrier, text);
    return { svg, line, carrier, textPath, mirror: sideIndex === 1 ? -1 : 1 };
  });
  if (!sides.length) return;

  // Length of one "velmora · trisk · ..." cycle, so the drift can loop without a jump.
  const cycleText = `${FLASHCARDS.map(([word]) => word).join(" · ")} · `;
  const probe = document.createElementNS(SVG_NS, "text");
  probe.textContent = cycleText;
  sides[0].svg.appendChild(probe);
  const cycleLength = probe.getComputedTextLength() || 400;
  probe.remove();

  function wavePath(height, centre, amplitude, wavelength, phase, mirror) {
    const points = [];
    for (let y = -40; y <= height + 40; y += 6) {
      const x = centre + mirror * (amplitude * Math.sin((y / wavelength) * Math.PI * 2 + phase)
        + amplitude * 0.35 * Math.sin((y / wavelength) * Math.PI * 5.3 + phase * 1.7));
      points.push(`${x.toFixed(1)},${y}`);
    }
    return `M${points.join(" L")}`;
  }

  let phase = 0;
  let drift = 0;
  function draw() {
    const height = window.innerHeight;
    for (const side of sides) {
      side.svg.setAttribute("viewBox", `0 0 ${WIDTH} ${height}`);
      side.line.setAttribute("d", wavePath(height, WIDTH * 0.42, 16, 420, phase + 1.4, side.mirror));
      // The letter path runs bottom-up on the right edge so the letters face outwards.
      side.carrier.setAttribute("d", wavePath(height, WIDTH * 0.62, 22, 560, phase, side.mirror));
      side.textPath.setAttribute("startOffset", `${-cycleLength + (drift % cycleLength)}`);
    }
  }

  draw();
  window.addEventListener("resize", draw);
  if (REDUCED_MOTION) return;
  function frame() {
    phase += 0.003;
    drift += 0.25;
    draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// On GitHub Pages (user.github.io/repo) link to github.com/user/repo.
function setRepositoryLink() {
  const { hostname, pathname } = window.location;
  if (!hostname.endsWith("github.io")) return;
  const user = hostname.split(".")[0];
  const repository = pathname.split("/").filter(Boolean)[0] ?? `${user}.github.io`;
  document.getElementById("repo-link").href = `https://github.com/${user}/${repository}`;
}

startFlashcards();
startBrainWave();
startSideWaves();
setRepositoryLink();
