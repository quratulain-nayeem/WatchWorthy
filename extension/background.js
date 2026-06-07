let lastUrl = {};

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (!tab.url || !tab.url.includes("youtube.com/watch")) return;

  // Fire on URL change OR on complete, whichever comes first
  if (changeInfo.url || changeInfo.status === "complete") {
    const url = tab.url;
    if (lastUrl[tabId] === url) return; // same URL, ignore
    lastUrl[tabId] = url;
    chrome.tabs.sendMessage(tabId, { type: "VIDEO_CHANGED", url });
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  delete lastUrl[tabId];
});