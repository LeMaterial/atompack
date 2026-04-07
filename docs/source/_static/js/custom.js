/* Copyright 2025 Entalpic */

// Description: Custom JS for the siesta theme
/**
 * Custom JS for the siesta theme
 */

/**
 * Update the HTML of the parameter lists
 * This function is used to update the HTML of the parameter lists in the documentation
 * to make them more readable.
 * It adds a custom class to the parameter names so that they can be styled differently
 * from the rest of the text.
 */
const updateParamHTMLs = () => {
    const pyElements = document.querySelectorAll("dl.py");
    pyElements.forEach((element) => {
        if (element.classList.contains("class")) return;

        const h3s = [...element.querySelectorAll("h3")].filter(
            (h) =>
                h.innerText.toLowerCase().indexOf("parameter") >= 0 ||
                h.innerText.toLowerCase().indexOf("argument") >= 0 ||
                h.innerText.toLowerCase().indexOf("return") >= 0
        );
        for (const h3 of h3s) {
            if (h3) {
                const dl = h3.nextElementSibling;
                if (dl) {
                    const dts = dl.querySelectorAll("dt");
                    for (const dt of dts) {
                        const code = dt.innerHTML;
                        let prefix = code.split("<span")[0];
                        let sufffix = code.split("<span")[1] || "";
                        if (prefix.indexOf("<span") < 0) {
                            if (prefix.indexOf(":") >= 0 && !sufffix.length) {
                                sufffix = `>${prefix.split(":")[1]}</span>`;
                                prefix = prefix.split(":")[0];
                            }
                            dt.innerHTML = `<span class="custom-param-list-item">${prefix}</span><span class="classifier"${sufffix}`;
                        }
                    }
                }
            }
        }
    });
};

(async () => {})(
    document.addEventListener("DOMContentLoaded", () => {
        updateParamHTMLs();
    })
);
