/*
 * TinyLLM Chatbot - safe markdown rendering
 *
 * LLM output is untrusted: it can echo text from web pages, RAG documents,
 * news feeds or search results (prompt injection). marked.js passes raw HTML
 * and javascript: links straight through, so rendering its output with
 * innerHTML would execute attacker-controlled script in the chat page.
 *
 * This module renders markdown with raw HTML escaped (shown as text) and
 * link/image URLs limited to safe schemes. It must be loaded after marked.js.
 */
(function (root) {
    function escapeHtml(unsafe) {
        return String(unsafe)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Allow http(s), mailto, and scheme-less (relative / anchor) URLs only.
    function isSafeUrl(url) {
        // Browsers ignore ASCII whitespace and control chars inside a scheme
        // (e.g. "java\tscript:"), so strip them before checking.
        const normalized = String(url || "").replace(/[\u0000- \u007f]/g, "").toLowerCase();
        const scheme = normalized.match(/^([a-z][a-z0-9+.\-]*):/);
        if (!scheme) {
            return true;
        }
        return ["http", "https", "mailto"].indexOf(scheme[1]) !== -1;
    }

    function safeUrl(url) {
        return isSafeUrl(url) ? url : "#";
    }

    const safeExtension = {
        renderer: {
            // Raw HTML blocks and inline tags are displayed as text
            html(token) {
                return escapeHtml(typeof token === "string" ? token : token.text);
            }
        },
        walkTokens(token) {
            if ((token.type === "link" || token.type === "image") && !isSafeUrl(token.href)) {
                token.href = "#";
            }
        }
    };

    function renderMarkdownSafe(text) {
        return root.marked.parse(String(text || ""), {
            breaks: true,
            gfm: true,
            pedantic: false
        });
    }

    if (root.marked && typeof root.marked.use === "function") {
        root.marked.use(safeExtension);
    }

    root.escapeHtml = escapeHtml;
    root.safeUrl = safeUrl;
    root.renderMarkdownSafe = renderMarkdownSafe;
})(typeof window !== "undefined" ? window : this);
