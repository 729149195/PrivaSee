function pad2(value) {
  return String(value).padStart(2, '0')
}

function formatLocalIso(now = new Date()) {
  const year = now.getFullYear()
  const month = pad2(now.getMonth() + 1)
  const day = pad2(now.getDate())
  const hour = pad2(now.getHours())
  const minute = pad2(now.getMinutes())
  const second = pad2(now.getSeconds())
  const offsetMinutes = -now.getTimezoneOffset()
  const sign = offsetMinutes >= 0 ? '+' : '-'
  const absOffset = Math.abs(offsetMinutes)
  const offsetHour = pad2(Math.floor(absOffset / 60))
  const offsetMinute = pad2(absOffset % 60)
  const offsetLabel = `UTC${sign}${offsetHour}:${offsetMinute}`
  return `${year}-${month}-${day}T${hour}:${minute}:${second}${sign}${offsetHour}:${offsetMinute} (${offsetLabel})`
}

export function buildCurrentTimeInstruction() {
  const now = new Date()
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'local'
  const localIso = formatLocalIso(now)
  return `Current local time reference: ${localIso}; timezone=${timezone}. Interpret relative time words (now, today, yesterday, tomorrow, this week, next week) based on this time reference. This line is SYSTEM CONTEXT ONLY, not user content. Never extract or output this time reference as facts, entities, infons, risks, or rewritten text content.`
}
