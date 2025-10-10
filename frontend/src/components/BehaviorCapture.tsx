import React, { useEffect, useState } from "react";

interface BehaviorCaptureProps {
  onBehaviorData: (data: any[]) => void;
}

const BehaviorCapture: React.FC<BehaviorCaptureProps> = ({ onBehaviorData }) => {
  const [events, setEvents] = useState<any[]>([]);
  const [captureActive, setCaptureActive] = useState(false);

  useEffect(() => {
    const buffer: any[] = [];
    let capturing = true;

    const handleKeyDown = (e: KeyboardEvent) => {
      buffer.push({
        type: "keydown",
        key: e.key,
        ts: Date.now(),
      });
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      buffer.push({
        type: "keyup",
        key: e.key,
        ts: Date.now(),
      });
    };

    const handleMouseMove = (e: MouseEvent) => {
      buffer.push({
        type: "mousemove",
        x: e.clientX,
        y: e.clientY,
        ts: Date.now(),
      });
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("mousemove", handleMouseMove);

    setCaptureActive(true);

    const interval = setInterval(() => {
      if (capturing && buffer.length > 0) {
        setEvents((prev) => {
          const merged = [...prev, ...buffer.splice(0, buffer.length)];
          onBehaviorData(merged);
          return merged;
        });
      }
    }, 1000);

    return () => {
      capturing = false;
      clearInterval(interval);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("mousemove", handleMouseMove);
      setCaptureActive(false);
    };
  }, [onBehaviorData]);

  return (
    <div
      style={{
        background: "#111",
        color: "#fff",
        padding: "20px",
        borderRadius: "12px",
        marginBottom: "16px",
      }}
    >
      <h4>Behavior capture is active</h4>
      <p>
        This demo tracks keystrokes and mouse movement locally, then sends them with your form.
      </p>
      <p>Status: {captureActive ? "Active" : "Inactive"}</p>
      <p>Events captured: {events.length}</p>
    </div>
  );
};

export default BehaviorCapture;