"""Provide the readable stylesheet bundled with the browser dashboard."""

STYLE = """
:root
{
    --bg: #0b1118;
    --panel: #151e29;
    --line: #29384a;
    --text: #e9f0f7;
    --muted: #8fa2b7;
    --accent: #48b6ff;
    --danger: #ff5c6c;
    --ok: #4ade80;
    --warn: #fbbf24;
}
*
{
    box-sizing: border-box;
}
body
{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px system-ui, sans-serif;
}
header
{
    position: sticky;
    top: 0;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 20px;
    background: #0f1721;
    border-bottom: 1px solid var(--line);
}
h1
{
    font-size: 18px;
    margin: 0;
}
#status
{
    color: var(--muted);
}
main
{
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 16px;
    padding: 16px;
}
.card
{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 16px;
}
h2
{
    font-size: 15px;
    margin: 0 0 14px;
}
.sidebar
{
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.workspace
{
    display: grid;
    gap: 16px;
    min-width: 0;
}
.camera-row
{
    margin-top: 16px;
}
.camera-row h3
{
    font-size: 13px;
    margin: 0 0 10px;
}
.camera-grid
{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}
.camera-grid figure
{
    margin: 0;
    min-width: 0;
}
.camera-grid img
{
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: contain;
    background: #0e1620;
    border-radius: 6px;
}
.camera-grid figcaption
{
    margin-top: 6px;
    color: var(--muted);
    font-size: 12px;
}
.camera-grid figcaption .lens
{
    opacity: 0.7;
}
#camera-panel.has-sensors
{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(240px, 340px);
    gap: 16px;
    align-items: start;
}
.camera-feeds
{
    min-width: 0;
}
#sensor-panel
{
    display: flex;
    flex-direction: column;
    min-width: 0;
}
#sensor-panel[hidden]
{
    display: none;
}
#sensor-windows
{
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.sensor-tile
{
    background: #0e1620;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 10px;
    min-width: 0;
}
.sensor-head
{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 13px;
}
.sensor-updated
{
    color: var(--muted);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.sensor-table
{
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 12px;
}
.sensor-table td
{
    padding: 3px 4px;
    border-top: 1px solid var(--line);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: top;
}
.sensor-table td.sensor-label
{
    width: 34%;
    border-left: 3px solid var(--muted);
    padding-left: 6px;
    color: #c8d5e3;
}
.sensor-table td.sensor-value
{
    width: 26%;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
}
.sensor-table td.sensor-detail
{
    color: var(--muted);
    white-space: normal;
    overflow-wrap: anywhere;
}
.sensor-table tr.sensor-ok td.sensor-label
{
    border-left-color: var(--ok);
}
.sensor-table tr.sensor-ok td.sensor-value
{
    color: var(--ok);
}
.sensor-table tr.sensor-warn td.sensor-label
{
    border-left-color: var(--warn);
}
.sensor-table tr.sensor-warn td.sensor-value
{
    color: var(--warn);
}
.sensor-table tr.sensor-idle td
{
    color: var(--muted);
}
.drive-grid
{
    display: grid;
    grid-template-columns: repeat(3, 64px);
    gap: 8px;
    justify-content: center;
    margin: 14px 0;
}
button
{
    border: 1px solid #3b526a;
    background: #223044;
    color: var(--text);
    border-radius: 8px;
    padding: 11px;
    cursor: pointer;
}
button:hover
{
    background: #2b4057;
}
button.stop
{
    background: #4a2028;
    border-color: #883945;
}
button.home
{
    width: 100%;
    margin-top: 8px;
}
.speed label
{
    display: grid;
    grid-template-columns: 90px 1fr 48px;
    align-items: center;
    gap: 8px;
    margin: 8px 0;
    color: var(--muted);
}
#joint-groups
{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(390px, 1fr));
    gap: 16px;
}
.joint-row
{
    display: grid;
    grid-template-columns: 170px 1fr 72px;
    gap: 10px;
    align-items: center;
    margin: 8px 0;
}
.joint-row span
{
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #c8d5e3;
}
.joint-row span small
{
    color: #7f8ea0;
    font-size: 11px;
}
input[type=range]
{
    width: 100%;
    accent-color: var(--accent);
}
input[type=number]
{
    width: 72px;
    background: #0e1620;
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 5px;
}
.hint
{
    color: var(--muted);
    font-size: 12px;
    line-height: 1.5;
}
.field-label
{
    display: block;
    margin: 10px 0 4px;
    color: var(--muted);
    font-size: 12px;
}
select
{
    width: 100%;
    margin-top: 4px;
    background: #0e1620;
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 7px 8px;
    font: inherit;
}
select:focus
{
    outline: none;
    border-color: var(--accent);
}
.coord-row
{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 4px;
}
.coord-row.quat-row
{
    grid-template-columns: repeat(4, 1fr);
}
.ik-target-header
{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 10px;
    margin-bottom: 2px;
}
.coord-row label
{
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--muted);
}
.coord-row input[type=number]
{
    width: 100%;
}
.orient-check
{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 12px;
    color: var(--muted);
    cursor: pointer;
}
.orient-check input[type=checkbox]
{
    accent-color: var(--accent);
    cursor: pointer;
}
.ik-buttons
{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}
button.btn-primary
{
    background: #1b4b72;
    border-color: #3172aa;
    color: #e9f0f7;
    font-weight: 500;
}
button.btn-primary:hover
{
    background: #245e8e;
}
#ik-status
{
    margin-top: 10px;
    min-height: 28px;
}
@media (max-width: 1000px)
{
    .camera-grid
    {
        grid-template-columns: 1fr;
    }
    #camera-panel.has-sensors
    {
        grid-template-columns: minmax(0, 1fr);
    }
}
@media (max-width: 800px)
{
    main
    {
        grid-template-columns: 1fr;
    }
    #joint-groups
    {
        grid-template-columns: 1fr;
    }
}
"""
