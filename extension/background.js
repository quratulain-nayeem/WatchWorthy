chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Only fire when the URL changes and the tab is fully loaded
  if (
    changeInfo.status === "complete" &&
    tab.url &&
    tab.url.includes("youtube.com/watch")
  ) {
    chrome.tabs.sendMessage(tabId, { type: "VIDEO_CHANGED", url: tab.url });
  }
});