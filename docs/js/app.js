/**
 * Trading Dashboard - reads data.json pushed by bot
 */

const DATA_URL = 'data.json';
const REFRESH_MS = 60_000; // check every 60s

let equityChart = null;
let monthlyChart = null;

// ── Helpers ──
function $(id) { return document.getElementById(id); }

function fmt(n, decimals = 2) {
    if (n == null) return '--';
    return n.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function fmtUsd(n) {
    if (n == null) return '--';
    const sign = n >= 0 ? '+' : '-';
    return `${sign}$${fmt(Math.abs(n))}`;
}

function fmtPct(n) {
    if (n == null) return '--';
    const sign = n >= 0 ? '+' : '';
    return `${sign}${fmt(n, 1)}%`;
}

function pnlClass(n) {
    if (n == null) return '';
    return n >= 0 ? 'positive' : 'negative';
}

function timeAgo(isoStr) {
    if (!isoStr) return '--';
    const diff = (Date.now() - new Date(isoStr).getTime()) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

function stratLabel(s) {
    const map = { ichimoku: 'Ichimoku', mirror_short: 'Mirror', ma100: 'MA100' };
    return map[s] || s;
}

// ── Data Fetcher ──
async function fetchData() {
    try {
        const res = await fetch(DATA_URL + '?t=' + Date.now());
        if (!res.ok) throw new Error(res.status);
        return await res.json();
    } catch (e) {
        console.error('Fetch failed:', e);
        return null;
    }
}

// ── Renderers ──
function renderOverview(data) {
    const b = data.balance || {};
    $('totalBalance').textContent = `$${fmt(b.total)}`;

    const change = b.daily_change;
    const changePct = b.daily_change_pct;
    const changeEl = $('balanceChange');
    changeEl.textContent = change != null ? `${fmtUsd(change)} (${fmtPct(changePct)}) today` : '--';
    changeEl.className = `card-sub ${pnlClass(change)}`;

    const upnl = b.unrealized_pnl;
    $('unrealizedPnl').textContent = upnl != null ? fmtUsd(upnl) : '--';
    $('unrealizedPnl').className = `card-value ${pnlClass(upnl)}`;
    const upnlPct = b.total && upnl ? (upnl / b.total * 100) : null;
    $('unrealizedPct').textContent = upnlPct != null ? `${fmtPct(upnlPct)} of balance` : '--';

    // Today's trades
    const today = new Date().toISOString().slice(0, 10);
    const todayTrades = (data.trades || []).filter(t =>
        t.closed_at && t.closed_at.startsWith(today)
    );
    $('todayTrades').textContent = todayTrades.length;
    const todayPnlVal = todayTrades.reduce((s, t) => s + (t.pnl_usd || 0), 0);
    const tdPnl = $('todayPnl');
    tdPnl.textContent = fmtUsd(todayPnlVal);
    tdPnl.className = `card-sub ${pnlClass(todayPnlVal)}`;

    // Overall stats
    const allTrades = data.trades || [];
    const wins = allTrades.filter(t => (t.pnl_usd || 0) > 0).length;
    const wr = allTrades.length > 0 ? (wins / allTrades.length * 100) : null;
    $('winRate').textContent = wr != null ? `${fmt(wr, 1)}%` : '--';

    const grossProfit = allTrades.reduce((s, t) => s + Math.max(0, t.pnl_usd || 0), 0);
    const grossLoss = Math.abs(allTrades.reduce((s, t) => s + Math.min(0, t.pnl_usd || 0), 0));
    const pf = grossLoss > 0 ? grossProfit / grossLoss : null;
    $('profitFactor').textContent = pf != null ? `PF ${fmt(pf)}` : '--';

    // Mode badge
    const badge = $('modeBadge');
    if (data.mode === 'paper') {
        badge.textContent = 'PAPER';
        badge.classList.add('paper');
    } else {
        badge.textContent = 'LIVE';
        badge.classList.remove('paper');
    }

    // Update time
    $('updateTime').textContent = data.updated_at ? timeAgo(data.updated_at) : '--';
    const dot = $('statusDot');
    if (!data.updated_at) {
        dot.className = 'status-dot offline';
    } else {
        const age = (Date.now() - new Date(data.updated_at).getTime()) / 1000;
        dot.className = age > 600 ? 'status-dot stale' : 'status-dot';
    }
}

function renderStrategies(data) {
    const strategies = data.strategies || {};

    for (const [key, conf] of Object.entries({
        ichimoku: { name: 'ichi', trades: data.trades },
        mirror_short: { name: 'mirror', trades: data.trades },
        ma100: { name: 'ma100', trades: data.trades },
    })) {
        const s = strategies[key] || {};
        const prefix = conf.name;

        // Status
        const statusEl = $(`${prefix}Status`);
        if (s.running) {
            statusEl.textContent = 'RUNNING';
            statusEl.className = 'strat-status running';
        } else {
            statusEl.textContent = 'STOPPED';
            statusEl.className = 'strat-status stopped';
        }

        // Stats from trades
        const stratTrades = (conf.trades || []).filter(t => t.strategy === key);
        const n = stratTrades.length;
        const wins = stratTrades.filter(t => (t.pnl_usd || 0) > 0).length;
        const totalPnl = stratTrades.reduce((s, t) => s + (t.pnl_usd || 0), 0);
        const gp = stratTrades.reduce((s, t) => s + Math.max(0, t.pnl_usd || 0), 0);
        const gl = Math.abs(stratTrades.reduce((s, t) => s + Math.min(0, t.pnl_usd || 0), 0));

        $(`${prefix}Trades`).textContent = n;
        $(`${prefix}WR`).textContent = n > 0 ? `${fmt(wins / n * 100, 1)}%` : '--';

        const pnlEl = $(`${prefix}Pnl`);
        pnlEl.textContent = fmtUsd(totalPnl);
        pnlEl.className = `stat-value ${pnlClass(totalPnl)}`;

        $(`${prefix}PF`).textContent = gl > 0 ? fmt(gp / gl) : '--';
    }
}

function renderPositions(data) {
    const positions = data.positions || [];
    $('posCount').textContent = positions.length;

    const tbody = $('positionsBody');
    if (positions.length === 0) {
        tbody.innerHTML = '<tr class="empty-row"><td colspan="9">No open positions</td></tr>';
        return;
    }

    tbody.innerHTML = positions.map(p => {
        const pnl = p.pnl_usd;
        const dcaInfo = p.dca_count ? `${p.dca_filled || 0}/${p.dca_count}` : '-';
        return `<tr>
            <td>${stratLabel(p.strategy)}</td>
            <td>${p.symbol || '--'}</td>
            <td>${(p.side || '--').toUpperCase()}</td>
            <td>$${fmt(p.entry_price, 4)}</td>
            <td>$${fmt(p.current_price, 4)}</td>
            <td>$${fmt(p.stop_loss, 4)}</td>
            <td>$${fmt(p.take_profit, 4)}</td>
            <td class="${pnlClass(pnl)}">${fmtUsd(pnl)}</td>
            <td>${dcaInfo}</td>
        </tr>`;
    }).join('');
}

function renderTrades(data) {
    const trades = (data.trades || []).slice(0, 50); // last 50
    const tbody = $('tradesBody');

    if (trades.length === 0) {
        tbody.innerHTML = '<tr class="empty-row"><td colspan="8">No trades yet</td></tr>';
        return;
    }

    tbody.innerHTML = trades.map(t => {
        const time = t.closed_at ? new Date(t.closed_at).toLocaleString() : '--';
        return `<tr>
            <td>${time}</td>
            <td>${stratLabel(t.strategy)}</td>
            <td>${t.symbol || '--'}</td>
            <td>${(t.side || '--').toUpperCase()}</td>
            <td>$${fmt(t.entry_price, 4)}</td>
            <td>$${fmt(t.exit_price, 4)}</td>
            <td class="${pnlClass(t.pnl_usd)}">${fmtUsd(t.pnl_usd)}</td>
            <td>${t.reason || '--'}</td>
        </tr>`;
    }).join('');
}

function renderDCA(data) {
    const dca = data.dca || {};
    const acc = dca.accumulated || {};

    const btc = acc.BTC || {};
    $('dcaBtcQty').textContent = btc.total_qty != null ? btc.total_qty.toFixed(8) : '0';
    $('dcaBtcInvested').textContent = btc.total_invested_usdt != null ? `$${fmt(btc.total_invested_usdt)}` : '$0';
    $('dcaBtcAvg').textContent = btc.total_qty > 0 ? `$${fmt(btc.total_invested_usdt / btc.total_qty, 0)}` : '--';
    $('dcaBtcCount').textContent = btc.buy_count || 0;

    const eth = acc.ETH || {};
    $('dcaEthQty').textContent = eth.total_qty != null ? eth.total_qty.toFixed(8) : '0';
    $('dcaEthInvested').textContent = eth.total_invested_usdt != null ? `$${fmt(eth.total_invested_usdt)}` : '$0';
    $('dcaEthAvg').textContent = eth.total_qty > 0 ? `$${fmt(eth.total_invested_usdt / eth.total_qty, 0)}` : '--';
    $('dcaEthCount').textContent = eth.buy_count || 0;

    const lastDca = dca.last_dca_time;
    const nextDca = dca.next_dca_time;
    let meta = '';
    if (lastDca) meta += `Last: ${timeAgo(lastDca)}`;
    if (nextDca) meta += ` | Next: ${new Date(nextDca).toLocaleString()}`;
    $('dcaMeta').textContent = meta || '--';
}

function renderEquityChart(data) {
    const history = data.equity_history || [];
    if (history.length === 0) return;

    const ctx = $('equityChart').getContext('2d');
    const labels = history.map(h => h.time);
    const values = history.map(h => h.balance);

    if (equityChart) {
        equityChart.data.labels = labels;
        equityChart.data.datasets[0].data = values;
        equityChart.update('none');
        return;
    }

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data: values,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.08)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#111119',
                    borderColor: '#1e1e2e',
                    borderWidth: 1,
                    titleColor: '#7a7a8e',
                    bodyColor: '#e4e4ef',
                    displayColors: false,
                    callbacks: {
                        label: ctx => `$${fmt(ctx.parsed.y)}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'day' },
                    grid: { color: 'rgba(30, 30, 46, 0.5)' },
                    ticks: { color: '#4a4a5e', font: { size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(30, 30, 46, 0.5)' },
                    ticks: {
                        color: '#4a4a5e',
                        font: { size: 10 },
                        callback: v => `$${v.toLocaleString()}`
                    }
                }
            }
        }
    });
}

function renderMonthlyChart(data) {
    const trades = data.trades || [];
    if (trades.length === 0) return;

    // Group by month
    const monthly = {};
    for (const t of trades) {
        if (!t.closed_at) continue;
        const month = t.closed_at.slice(0, 7); // YYYY-MM
        if (!monthly[month]) monthly[month] = { pnl: 0, count: 0, wins: 0 };
        monthly[month].pnl += t.pnl_usd || 0;
        monthly[month].count++;
        if ((t.pnl_usd || 0) > 0) monthly[month].wins++;
    }

    const months = Object.keys(monthly).sort();
    const pnls = months.map(m => monthly[m].pnl);
    const colors = pnls.map(p => p >= 0 ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)');

    const ctx = $('monthlyChart').getContext('2d');

    if (monthlyChart) {
        monthlyChart.data.labels = months;
        monthlyChart.data.datasets[0].data = pnls;
        monthlyChart.data.datasets[0].backgroundColor = colors;
        monthlyChart.update('none');
        return;
    }

    monthlyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: months,
            datasets: [{
                data: pnls,
                backgroundColor: colors,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#111119',
                    borderColor: '#1e1e2e',
                    borderWidth: 1,
                    titleColor: '#7a7a8e',
                    bodyColor: '#e4e4ef',
                    displayColors: false,
                    callbacks: {
                        label: ctx => {
                            const m = months[ctx.dataIndex];
                            const d = monthly[m];
                            return [
                                `PnL: ${fmtUsd(d.pnl)}`,
                                `Trades: ${d.count} (WR ${fmt(d.wins / d.count * 100, 0)}%)`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#4a4a5e', font: { size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(30, 30, 46, 0.5)' },
                    ticks: {
                        color: '#4a4a5e',
                        font: { size: 10 },
                        callback: v => `$${v.toLocaleString()}`
                    }
                }
            }
        }
    });
}

// ── Main ──
async function refresh() {
    const data = await fetchData();
    if (!data) return;

    renderOverview(data);
    renderStrategies(data);
    renderPositions(data);
    renderTrades(data);
    renderDCA(data);
    renderEquityChart(data);
    renderMonthlyChart(data);
}

// Initial load + auto refresh
refresh();
setInterval(refresh, REFRESH_MS);
